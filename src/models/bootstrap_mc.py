"""
bootstrap_mc.py — Module 8: Bootstrap Monte Carlo for regular sawtooth satellites.

Strategy:
1. Fit a Ridge regression baseline on the reset-free residual segments
2. Characterize the reset distribution from training history:
      - Inter-reset intervals (mean, std)
      - Reset magnitudes (mean, std)
3. Simulate BOOTSTRAP_N_SAMPLES possible futures:
      - Draw reset times from historical distribution
      - Draw reset magnitudes from historical distribution
      - Add to baseline prediction
4. Aggregate:
      - mean trajectory = ensemble average
      - std trajectory  = ensemble standard deviation (honest uncertainty)

This approach is "generative" in the competition's sense:
it synthesizes realistic future trajectories rather than producing
a single point prediction. The uncertainty is calibrated by the actual
historical variability in reset timing and magnitude.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import (
    BOOTSTRAP_N_SAMPLES,
    BOOTSTRAP_SEED,
    ORBITAL_PERIODS_HR,
    SOLAR_PERIOD_HR,
    PREDICTION_INTERVAL_MIN,
)
from src.reset_detector import ResetEvent, EventType, reset_statistics

logger = logging.getLogger(__name__)


class BootstrapMCModel:
    """
    Bootstrap Monte Carlo model for regular sawtooth satellites.

    Usage
    -----
    model = BootstrapMCModel(sat_id="E05", orbital_period_hr=14.083, constellation="GALILEO")
    model.fit(timestamps, residual_ns, resets)
    mean, std = model.predict(future_timestamps)
    """

    def __init__(
        self,
        sat_id: str,
        orbital_period_hr: float,
        constellation: str = "GPS",
        n_bootstrap: int = BOOTSTRAP_N_SAMPLES,
        seed: int = BOOTSTRAP_SEED,
    ):
        self.sat_id = sat_id
        self.orbital_period_hr = orbital_period_hr
        self.constellation = constellation
        self.n_bootstrap = n_bootstrap
        self.rng = np.random.default_rng(seed)

        self._ridge = None
        self._scaler = None
        self._t0: Optional[pd.Timestamp] = None
        self._y_mean: float = 0.0
        self._reset_stats: dict = {}
        self._training_duration_hr: float = 0.0
        self._last_reset_hr: float = 0.0   # Hours since t0 of last training reset

        logger.debug(
            f"[{sat_id}] BootstrapMCModel initialized | "
            f"period={orbital_period_hr:.3f}hr | n_bootstrap={n_bootstrap}"
        )

    def fit(
        self,
        timestamps: pd.DatetimeIndex,
        residual_ns: pd.Series,
        resets: List[ResetEvent],
    ) -> "BootstrapMCModel":
        """
        Fit the baseline model and characterize the reset distribution.

        Parameters
        ----------
        timestamps : pd.DatetimeIndex
        residual_ns : pd.Series — detrended residual in ns
        resets : List[ResetEvent] — detected resets (from reset_detector.py)
        """
        valid_mask = residual_ns.notna()
        ts_valid = timestamps[valid_mask]
        y_valid  = residual_ns[valid_mask].values

        self._t0 = ts_valid[0]
        t_hours = np.array([(t - self._t0).total_seconds() / 3600.0 for t in ts_valid])
        self._training_duration_hr = float(t_hours[-1])

        self._y_mean = float(np.mean(y_valid))

        # --- Compute reset statistics for future simulation ---
        self._reset_stats = reset_statistics(resets)

        # Track when the last reset occurred (in hours since t0)
        real_resets = [e for e in resets if e.event_type == EventType.RESET]
        if real_resets:
            last_reset_ts = max(real_resets, key=lambda e: e.timestamp).timestamp
            self._last_reset_hr = (last_reset_ts - self._t0).total_seconds() / 3600.0
        else:
            self._last_reset_hr = 0.0

        # --- Fit baseline Ridge on reset-free segments ---
        # Mask out ±1 step around each reset to avoid contamination
        reset_mask = np.ones(len(ts_valid), dtype=bool)
        for reset_evt in real_resets:
            r_idx = reset_evt.index
            # Map reset index back to valid-only index space
            contaminated = np.where(
                (t_hours >= t_hours[max(0, r_idx-2)]) &
                (t_hours <= t_hours[min(len(t_hours)-1, r_idx+2)])
            )[0]
            reset_mask[contaminated] = False

        t_clean = t_hours[reset_mask]
        y_clean = y_valid[reset_mask]

        if len(t_clean) < 20:
            logger.warning(f"[{self.sat_id}] Too few clean points for baseline fit. Using all data.")
            t_clean = t_hours
            y_clean = y_valid

        X_clean = self._build_features(t_clean)
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_clean)
        self._ridge = Ridge(alpha=1.0)
        self._ridge.fit(X_scaled, y_clean - self._y_mean)

        # Fit quality
        y_pred_clean = self._ridge.predict(X_scaled) + self._y_mean
        residuals_fit = y_clean - y_pred_clean
        rmse = float(np.sqrt(np.mean(residuals_fit**2)))

        logger.info(
            f"[{self.sat_id}] BootstrapMC baseline RMSE={rmse:.3f} ns | "
            f"n_resets={self._reset_stats['n_resets']} | "
            f"mean_interval={self._reset_stats['mean_interval_hr']:.1f}hr"
        )
        return self

    def predict(
        self,
        future_timestamps: pd.DatetimeIndex,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate bootstrapped trajectory ensemble and return mean ± std.

        Parameters
        ----------
        future_timestamps : pd.DatetimeIndex — 96 future timestamps

        Returns
        -------
        mean : np.ndarray shape (96,) in ns
        std  : np.ndarray shape (96,) in ns
        """
        if self._ridge is None:
            raise RuntimeError(f"[{self.sat_id}] Model not fitted. Call fit() first.")

        n_future = len(future_timestamps)
        t_future_hr = np.array(
            [(t - self._t0).total_seconds() / 3600.0 for t in future_timestamps]
        )

        # --- Baseline prediction (deterministic part) ---
        X_future = self._build_features(t_future_hr)
        X_scaled = self._scaler.transform(X_future)
        baseline = self._ridge.predict(X_scaled) + self._y_mean

        # --- Bootstrap: simulate N futures with random resets ---
        ensemble = np.zeros((self.n_bootstrap, n_future))

        for b in range(self.n_bootstrap):
            reset_contribution = self._simulate_resets(t_future_hr)
            ensemble[b] = baseline + reset_contribution

        mean_pred = np.mean(ensemble, axis=0)
        std_pred  = np.std(ensemble,  axis=0)

        logger.debug(
            f"[{self.sat_id}] BootstrapMC predict: "
            f"mean=[{mean_pred.min():.2f}, {mean_pred.max():.2f}] ns | "
            f"std=[{std_pred.min():.2f}, {std_pred.max():.2f}] ns"
        )

        return mean_pred, std_pred

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------

    def _build_features(self, t_hours: np.ndarray) -> np.ndarray:
        """Build feature matrix for Ridge baseline (without reset events)."""
        features = []
        # Polynomial (degree 3)
        for d in range(4):
            features.append(t_hours ** d)
        # Orbital harmonic
        omega_orb = 2 * np.pi / self.orbital_period_hr
        features.append(np.sin(omega_orb * t_hours))
        features.append(np.cos(omega_orb * t_hours))
        # Solar harmonic
        omega_sol = 2 * np.pi / SOLAR_PERIOD_HR
        features.append(np.sin(omega_sol * t_hours))
        features.append(np.cos(omega_sol * t_hours))
        return np.column_stack(features)

    def _simulate_resets(self, t_future_hr: np.ndarray) -> np.ndarray:
        """
        Simulate one possible sequence of resets in the prediction window.

        Strategy:
        - Draw the first reset time based on expected interval since last reset
        - Draw subsequent reset times from historical interval distribution
        - Draw magnitudes from historical magnitude distribution
        - Return the cumulative sawtooth contribution
        """
        stats = self._reset_stats
        if stats["n_resets"] == 0 or stats["mean_interval_hr"] == float("inf"):
            return np.zeros(len(t_future_hr))

        mean_interval = stats["mean_interval_hr"]
        std_interval  = max(stats["std_interval_hr"], 0.1)
        mean_mag      = stats["mean_magnitude_ns"]
        std_mag       = max(stats["std_magnitude_ns"], 0.1)

        # Time since last reset at start of prediction window
        time_of_first_future = t_future_hr[0]
        time_since_last_reset = time_of_first_future - self._last_reset_hr

        # Expected time to NEXT reset
        expected_next = mean_interval - time_since_last_reset
        if expected_next <= 0:
            expected_next = mean_interval  # Already overdue

        # Draw first reset time (relative to start of future window)
        next_reset_dt = max(
            self.rng.normal(expected_next, std_interval * 0.5),
            1.0  # Minimum 1 hour before first reset
        )

        contribution = np.zeros(len(t_future_hr))
        current_level = 0.0
        current_hr = t_future_hr[0]

        while True:
            reset_hr = current_hr + next_reset_dt
            if reset_hr > t_future_hr[-1]:
                break

            # Magnitude (signed; typically negative for sawtooth reset)
            mag = self.rng.normal(mean_mag, std_mag)

            # Apply reset: add magnitude at all future points after this time
            apply_mask = t_future_hr >= reset_hr
            contribution[apply_mask] += mag
            current_level += mag

            current_hr = reset_hr
            next_reset_dt = max(
                self.rng.normal(mean_interval, std_interval),
                1.0
            )

        return contribution
