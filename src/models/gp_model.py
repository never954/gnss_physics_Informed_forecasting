"""
gp_model.py — Module 7: 5-Kernel Gaussian Process for clean satellites.

Kernel composition (additive):
    K = K_rbf(l=clock_range)                   ← Slow clock drift
      + K_periodic_1(p≈orbital_period)          ← Orbital harmonic
      + K_periodic_2(p≈24hr, solar)             ← Solar/diurnal harmonic
      + K_matern(l=4hr, ν=1.5)                  ← Short-range wiggles
      + K_noise(σ²)                             ← Observation noise

Clock-type-aware initialization:
    H-maser  → long RBF range (80hr) — very stable
    Rubidium → medium RBF range (50hr)
    Cesium   → short RBF range (30hr) — noisier

Orbital period is learnable within ±30% of the constellation-specific value.
This anchors the kernel to physics while allowing fine-tuning from data.

Uses GPyTorch for GPU-optional training (falls back gracefully to CPU).
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import (
    GP_TRAINING_ITERATIONS,
    GP_LEARNING_RATE,
    GP_NOISE_INIT,
    ORBITAL_PERIOD_TOLERANCE,
    SOLAR_PERIOD_HR,
    SOLAR_PERIOD_MIN_HR,
    SOLAR_PERIOD_MAX_HR,
    SHORT_RANGE_MATERN_LENGTH_HR,
    PREDICTION_INTERVAL_MIN,
    get_clock_type,
    CLOCK_KERNEL_CONFIGS,
)

logger = logging.getLogger(__name__)

# Suppress gpytorch/torch optimization warnings in production
warnings.filterwarnings("ignore", category=RuntimeWarning, module="gpytorch")


def _try_import_gpytorch():
    try:
        import gpytorch
        return gpytorch
    except ImportError:
        raise ImportError(
            "gpytorch is required for GP models. "
            "Install with: pip install gpytorch"
        )


# ---------------------------------------------------------------------------
# GPyTorch model definition
# ---------------------------------------------------------------------------

class _GNSSGPModel(torch.nn.Module if False else object):
    pass  # Replaced by ExactGP below


def _make_gnss_gp_model(train_x, train_y, likelihood, orbital_period_hr, clock_type, gpytorch):
    """
    Factory: returns an ExactGP subclass with 5-kernel composition.
    Defined as a factory function so we can subclass gpytorch.models.ExactGP
    dynamically (avoids circular import issues).
    """
    gp = gpytorch
    clock_cfg = CLOCK_KERNEL_CONFIGS.get(clock_type, CLOCK_KERNEL_CONFIGS["UNKNOWN"])
    rbf_init = clock_cfg.rbf_range_hr
    rbf_lo, rbf_hi = clock_cfg.rbf_range_bounds
    p_lo = orbital_period_hr * (1 - ORBITAL_PERIOD_TOLERANCE)
    p_hi = orbital_period_hr * (1 + ORBITAL_PERIOD_TOLERANCE)

    class GNSSExactGP(gp.models.ExactGP):
        def __init__(self):
            super().__init__(train_x, train_y, likelihood)
            # 1. Slow drift (RBF)
            self.rbf_kernel = gp.kernels.RBFKernel()
            self.rbf_kernel.lengthscale = rbf_init
            self.rbf_kernel.register_constraint("raw_lengthscale", gp.constraints.Interval(rbf_lo, rbf_hi))
            # 2. Orbital periodic
            self.orbital_kernel = gp.kernels.PeriodicKernel()
            self.orbital_kernel.period_length = orbital_period_hr
            self.orbital_kernel.register_constraint("raw_period_length", gp.constraints.Interval(p_lo, p_hi))
            # 3. Solar periodic
            self.solar_kernel = gp.kernels.PeriodicKernel()
            self.solar_kernel.period_length = SOLAR_PERIOD_HR
            self.solar_kernel.register_constraint("raw_period_length", gp.constraints.Interval(SOLAR_PERIOD_MIN_HR, SOLAR_PERIOD_MAX_HR))
            # 4. Short-range Matérn ν=1.5
            self.matern_kernel = gp.kernels.MaternKernel(nu=1.5)
            self.matern_kernel.lengthscale = SHORT_RANGE_MATERN_LENGTH_HR
            self.matern_kernel.register_constraint("raw_lengthscale", gp.constraints.Interval(0.5, 12.0))
            # Composite kernel
            self.covar_module = (
                self.rbf_kernel + self.orbital_kernel + self.solar_kernel + self.matern_kernel
            )
            self.mean_module = gp.means.ConstantMean()

        def forward(self, x):
            mean_x  = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gp.distributions.MultivariateNormal(mean_x, covar_x)

    return GNSSExactGP()


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class GPModel:
    """
    5-kernel GP model for clean satellites.

    Usage
    -----
    model = GPModel(sat_id="G01", orbital_period_hr=11.9667)
    model.fit(timestamps, residual_ns)
    mean, std = model.predict(future_timestamps)
    """

    def __init__(
        self,
        sat_id: str,
        orbital_period_hr: float,
        clock_type: Optional[str] = None,
        n_iterations: int = GP_TRAINING_ITERATIONS,
        learning_rate: float = GP_LEARNING_RATE,
        use_gpu: bool = False,
    ):
        self.sat_id = sat_id
        self.orbital_period_hr = orbital_period_hr
        self.clock_type = clock_type or get_clock_type(sat_id)
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

        self._model = None
        self._likelihood = None
        self._t0 = None          # Reference time for normalization
        self._t_scale = None     # Time scale factor (hours → normalized)
        self._y_mean = None
        self._y_std = None

        logger.debug(
            f"[{sat_id}] GPModel initialized | "
            f"period={orbital_period_hr:.3f}hr | "
            f"clock={self.clock_type} | device={self.device}"
        )

    def fit(
        self,
        timestamps: pd.DatetimeIndex,
        residual_ns: pd.Series,
        verbose: bool = False,
    ) -> "GPModel":
        """
        Train the GP on the residual time series.

        Parameters
        ----------
        timestamps : pd.DatetimeIndex
            Training timestamps.
        residual_ns : pd.Series
            Detrended residual in nanoseconds.
        verbose : bool
            Print loss every 10 iterations.

        Returns
        -------
        self (for chaining)
        """
        gpytorch = _try_import_gpytorch()

        # Drop NaNs
        valid_mask = residual_ns.notna()
        ts_valid = timestamps[valid_mask]
        y_valid  = residual_ns[valid_mask].values

        if len(y_valid) < 20:
            raise ValueError(f"[{self.sat_id}] Too few valid points for GP fitting: {len(y_valid)}")

        # --- Normalize time to hours, centred at 0 ---
        self._t0 = ts_valid[0]
        t_hours = np.array([(t - self._t0).total_seconds() / 3600.0 for t in ts_valid])
        self._t_scale = float(t_hours[-1] - t_hours[0]) or 1.0
        t_norm = t_hours  # Keep in hours; GP kernels are parameterized in hours too

        # --- Normalize target (zero mean, unit std) ---
        self._y_mean = float(np.mean(y_valid))
        self._y_std  = float(np.std(y_valid)) or 1.0
        y_norm = (y_valid - self._y_mean) / self._y_std

        # Tensors
        train_x = torch.tensor(t_norm, dtype=torch.float32).to(self.device)
        train_y = torch.tensor(y_norm, dtype=torch.float32).to(self.device)

        # Initialize likelihood and model
        self._likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self._likelihood.noise = GP_NOISE_INIT / (self._y_std ** 2)  # Normalize noise too

        self._model = _make_gnss_gp_model(
            train_x=train_x,
            train_y=train_y,
            likelihood=self._likelihood,
            orbital_period_hr=self.orbital_period_hr,
            clock_type=self.clock_type,
            gpytorch=gpytorch,
        )

        self._model = self._model.to(self.device)
        self._likelihood = self._likelihood.to(self.device)

        # Train mode
        self._model.train()
        self._likelihood.train()

        optimizer = torch.optim.Adam(
            list(self._model.parameters()) + list(self._likelihood.parameters()),
            lr=self.learning_rate,
        )

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self._likelihood, self._model)

        losses = []
        for i in range(self.n_iterations):
            optimizer.zero_grad()
            output = self._model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if verbose and (i + 1) % 25 == 0:
                print(f"[{self.sat_id}] GP iter {i+1}/{self.n_iterations} | loss={loss.item():.4f}")

        final_loss = losses[-1]
        logger.info(
            f"[{self.sat_id}] GP trained | "
            f"loss={final_loss:.4f} | "
            f"n_train={len(y_norm)} | "
            f"orbital_period_learned="
            f"{self._model.orbital_kernel.period_length.item():.3f}hr"
        )

        return self

    def predict(
        self,
        future_timestamps: pd.DatetimeIndex,
        n_samples: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions for future timestamps.

        Parameters
        ----------
        future_timestamps : pd.DatetimeIndex
            96 future timestamps for Day 8 prediction.
        n_samples : int
            If > 0, also return posterior samples (for ensemble diversity).
            If 0, return (mean, std) only.

        Returns
        -------
        mean : np.ndarray shape (96,)  — predicted residual in nanoseconds
        std  : np.ndarray shape (96,)  — posterior standard deviation in nanoseconds
        """
        if self._model is None:
            raise RuntimeError(f"[{self.sat_id}] GP model not trained. Call fit() first.")

        gpytorch = _try_import_gpytorch()

        # Convert future timestamps to normalized hours
        t_future_hours = np.array(
            [(t - self._t0).total_seconds() / 3600.0 for t in future_timestamps]
        )
        test_x = torch.tensor(t_future_hours, dtype=torch.float32).to(self.device)

        # Eval mode
        self._model.eval()
        self._likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self._likelihood(self._model(test_x))
            mean_norm = observed_pred.mean.cpu().numpy()
            std_norm  = observed_pred.stddev.cpu().numpy()

        # Denormalize
        mean_ns = mean_norm * self._y_std + self._y_mean
        std_ns  = std_norm  * self._y_std

        logger.debug(
            f"[{self.sat_id}] GP predict: "
            f"mean=[{mean_ns.min():.2f}, {mean_ns.max():.2f}] ns | "
            f"std=[{std_ns.min():.2f}, {std_ns.max():.2f}] ns"
        )

        return mean_ns, std_ns

    @property
    def learned_orbital_period_hr(self) -> Optional[float]:
        """Return the orbital period learned by the GP (post-training)."""
        if self._model is not None:
            try:
                return float(self._model.orbital_kernel.period_length.item())
            except Exception:
                return None
        return None
