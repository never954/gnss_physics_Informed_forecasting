"""
matern_fallback.py — Module 10: Matérn-only GP fallback.

Used when a satellite doesn't fit any of the three primary classifications
(e.g., too few data points to classify, unknown constellation with no FFT peak,
or any model training failure).

Why Matérn ν=2.5 (not RBF, not ν=1.5)?
- Matérn ν=2.5: twice-differentiable (smooth) but not infinitely smooth
  → realistic for real-world time series
- RBF: infinitely smooth → can overfit and produce overconfident predictions
- Matérn ν=1.5: once-differentiable → too rough for 24hr extrapolation

This model is simple and always produces valid predictions.
It may not capture the orbital/solar periodicity as well as the 5-kernel GP,
but it never fails, and its uncertainty is honest (wide when data is sparse).
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
    MATERN_NU,
    MATERN_FALLBACK_LENGTH_HR,
    PREDICTION_INTERVAL_MIN,
)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="gpytorch")


def _try_import_gpytorch():
    try:
        import gpytorch
        return gpytorch
    except ImportError:
        raise ImportError("gpytorch is required. Install with: pip install gpytorch")


def _make_fallback_gp(train_x, train_y, gpytorch):
    """
    Factory: Matérn ν=2.5 ExactGP fallback model.
    """
    gp = gpytorch

    class FallbackExactGP(gp.models.ExactGP):
        def __init__(self, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module  = gp.means.ConstantMean()
            self.covar_module = gp.kernels.MaternKernel(nu=MATERN_NU)
            self.covar_module.lengthscale = MATERN_FALLBACK_LENGTH_HR
            self.covar_module.register_constraint(
                "raw_lengthscale",
                gp.constraints.Interval(1.0, 50.0),
            )

        def forward(self, x):
            return gp.distributions.MultivariateNormal(
                self.mean_module(x), self.covar_module(x)
            )

    return FallbackExactGP


class MaternFallbackModel:
    """
    Simple Matérn ν=2.5 GP fallback for unclassified satellites.

    Usage
    -----
    model = MaternFallbackModel(sat_id="UNKNOWN_01")
    model.fit(timestamps, residual_ns)
    mean, std = model.predict(future_timestamps)
    """

    def __init__(
        self,
        sat_id: str,
        n_iterations: int = min(GP_TRAINING_ITERATIONS, 100),  # Faster fallback
        learning_rate: float = GP_LEARNING_RATE,
        use_gpu: bool = False,
    ):
        self.sat_id = sat_id
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

        self._model = None
        self._likelihood = None
        self._t0 = None
        self._y_mean = None
        self._y_std = None

    def fit(
        self,
        timestamps: pd.DatetimeIndex,
        residual_ns: pd.Series,
        verbose: bool = False,
    ) -> "MaternFallbackModel":
        """Train single Matérn GP."""
        gpytorch = _try_import_gpytorch()

        valid_mask = residual_ns.notna()
        ts_valid = timestamps[valid_mask]
        y_valid  = residual_ns[valid_mask].values

        if len(y_valid) < 10:
            raise ValueError(f"[{self.sat_id}] Fallback: too few points ({len(y_valid)})")

        self._t0     = ts_valid[0]
        t_hours = np.array([(t - self._t0).total_seconds() / 3600.0 for t in ts_valid])
        self._y_mean = float(np.mean(y_valid))
        self._y_std  = float(np.std(y_valid)) or 1.0
        y_norm = (y_valid - self._y_mean) / self._y_std

        train_x = torch.tensor(t_hours, dtype=torch.float32).to(self.device)
        train_y = torch.tensor(y_norm,  dtype=torch.float32).to(self.device)

        self._likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        FallbackGP = _make_fallback_gp(train_x, train_y, gpytorch)
        self._model = FallbackGP(self._likelihood).to(self.device)

        self._model.train()
        self._likelihood.train()

        optimizer = torch.optim.Adam(
            list(self._model.parameters()) + list(self._likelihood.parameters()),
            lr=self.learning_rate,
        )
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self._likelihood, self._model)

        for i in range(self.n_iterations):
            optimizer.zero_grad()
            output = self._model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            if verbose and (i + 1) % 20 == 0:
                print(f"[{self.sat_id}] Fallback iter {i+1} | loss={loss.item():.4f}")

        logger.info(
            f"[{self.sat_id}] MaternFallback trained | "
            f"loss={loss.item():.4f} | n_train={len(y_norm)}"
        )
        return self

    def predict(
        self,
        future_timestamps: pd.DatetimeIndex,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mean_ns, std_ns) for future timestamps."""
        if self._model is None:
            raise RuntimeError(f"[{self.sat_id}] not trained. Call fit() first.")

        gpytorch = _try_import_gpytorch()

        t_future_hr = np.array(
            [(t - self._t0).total_seconds() / 3600.0 for t in future_timestamps]
        )
        test_x = torch.tensor(t_future_hr, dtype=torch.float32).to(self.device)

        self._model.eval()
        self._likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self._likelihood(self._model(test_x))
            mean_norm = observed_pred.mean.cpu().numpy()
            std_norm  = observed_pred.stddev.cpu().numpy()

        mean_ns = mean_norm * self._y_std + self._y_mean
        std_ns  = std_norm  * self._y_std

        logger.debug(
            f"[{self.sat_id}] Fallback predict: "
            f"mean=[{mean_ns.min():.2f}, {mean_ns.max():.2f}] ns"
        )
        return mean_ns, std_ns
