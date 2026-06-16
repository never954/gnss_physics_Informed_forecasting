"""
detrend.py — Module 5: Physics-informed detrending.

Strips the predictable physics from the SISE time series:
    1. Polynomial clock drift (degree 3 Ridge regression)
    2. Constellation-specific orbital harmonic (sine + cosine at orbital period)
    3. Solar/diurnal harmonic (sine + cosine at 24 hours)

What remains (the residual) is fed to the ML models.
At prediction time, the trend is re-added to model outputs.

Key design choices:
- Ridge regression (not OLS) to avoid overfitting on 7 days of data
- Exact orbital periods per constellation (not rounded to nearest hour)
- Separate solar term because GEO and MEO have different dominant components
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    ORBITAL_PERIODS_HR,
    SOLAR_PERIOD_HR,
    PREDICTION_INTERVAL_MIN,
    DATA_PROCESSED_DIR,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class DetrendResult:
    sat_id: str
    constellation: str
    orbital_period_hr: float

    # Input
    timestamps: pd.DatetimeIndex     # All training timestamps
    sise_ns: pd.Series               # Original SISE (ns)

    # Fit components (on training data)
    trend_poly: np.ndarray           # Polynomial component
    trend_orbital: np.ndarray        # Orbital harmonic component
    trend_solar: np.ndarray          # Solar harmonic component
    trend_total: np.ndarray          # Sum of all trend components
    residual: pd.Series              # sise_ns − trend_total

    # Model coefficients (for extrapolation)
    ridge_model: Ridge
    scaler: StandardScaler
    poly_degree: int
    orbital_period_hr_fit: float     # May differ slightly if adjusted

    def predict_trend(self, future_timestamps: pd.DatetimeIndex) -> np.ndarray:
        """
        Extrapolate the physics trend to future timestamps.
        Called after model predicts the residual — trend is re-added.

        Parameters
        ----------
        future_timestamps : pd.DatetimeIndex
            96 future timestamps (Day 8).

        Returns
        -------
        np.ndarray : trend values in nanoseconds
        """
        t_future = _timestamps_to_hours(future_timestamps, self.timestamps[0])
        X_future = _build_feature_matrix(
            t_future,
            self.poly_degree,
            self.orbital_period_hr_fit,
            SOLAR_PERIOD_HR,
        )
        X_future_scaled = self.scaler.transform(X_future)
        return self.ridge_model.predict(X_future_scaled)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detrend(
    sat_id: str,
    sise_series: pd.Series,
    constellation: str,
    orbital_period_hr: Optional[float] = None,
    poly_degree: int = 3,
    ridge_alpha: float = 1.0,
    save_to_disk: bool = False,
    output_dir: str = DATA_PROCESSED_DIR,
) -> DetrendResult:
    """
    Remove physics trend from a satellite's SISE time series.

    Parameters
    ----------
    sat_id : str
        Satellite PRN.
    sise_series : pd.Series
        SISE time series (ns), indexed by DatetimeIndex.
    constellation : str
        Constellation name for period lookup (e.g., "GPS", "GALILEO").
    orbital_period_hr : float, optional
        Override the constellation default orbital period.
    poly_degree : int
        Polynomial degree for clock drift term (default 3).
    ridge_alpha : float
        Ridge regularization strength (default 1.0).
    save_to_disk : bool
        If True, save residual CSV to output_dir.
    output_dir : str
        Directory for saving processed residuals.

    Returns
    -------
    DetrendResult
        Contains residual, trend components, and a predict_trend() method.
    """
    # Drop NaNs for fitting (but keep index for residual alignment)
    valid_mask = sise_series.notna()
    sise_valid = sise_series[valid_mask]

    if len(sise_valid) < 50:
        raise ValueError(f"[{sat_id}] Too few valid points ({len(sise_valid)}) for detrending.")

    timestamps = sise_series.index
    t_hours = _timestamps_to_hours(timestamps, timestamps[0])
    t_valid = t_hours[valid_mask]

    # --- Resolve orbital period ---
    if orbital_period_hr is None:
        orbital_period_hr = ORBITAL_PERIODS_HR.get(constellation, ORBITAL_PERIODS_HR["UNKNOWN"])
    logger.debug(f"[{sat_id}] Orbital period: {orbital_period_hr:.4f} hr")

    # --- Build feature matrix ---
    X = _build_feature_matrix(t_valid, poly_degree, orbital_period_hr, SOLAR_PERIOD_HR)

    # --- Fit Ridge regression ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ridge = Ridge(alpha=ridge_alpha, fit_intercept=True)
    ridge.fit(X_scaled, sise_valid.values)

    # --- Compute trend components ---
    # Build full feature matrix (including NaN positions — for alignment)
    X_full = _build_feature_matrix(t_hours, poly_degree, orbital_period_hr, SOLAR_PERIOD_HR)
    X_full_scaled = scaler.transform(X_full)

    trend_full = ridge.predict(X_full_scaled)

    # Decompose into components for interpretability
    n_poly = poly_degree + 1
    n_orbital = 2   # sin + cos
    n_solar = 2     # sin + cos

    # Reconstruct each component by zeroing out other feature groups
    def predict_component(feature_slice: slice) -> np.ndarray:
        coef_subset = np.zeros_like(ridge.coef_)
        coef_subset[feature_slice] = ridge.coef_[feature_slice]
        return X_full_scaled @ coef_subset + ridge.intercept_ / 3  # Split intercept evenly

    trend_poly    = _predict_component_safe(ridge, scaler, X_full, slice(0, n_poly))
    trend_orbital = _predict_component_safe(ridge, scaler, X_full, slice(n_poly, n_poly + n_orbital))
    trend_solar   = _predict_component_safe(ridge, scaler, X_full, slice(n_poly + n_orbital, n_poly + n_orbital + n_solar))

    # Residual = SISE - full trend (NaN where original was NaN)
    residual_vals = sise_series.values.copy().astype(float) - trend_full
    residual = pd.Series(residual_vals, index=timestamps, name="residual_ns")

    # --- Quality check ---
    r2 = 1.0 - (np.var(sise_valid.values - ridge.predict(X_scaled)) / np.var(sise_valid.values))
    logger.info(
        f"[{sat_id}] Detrend R²={r2:.4f} | "
        f"Residual std={residual.std():.3f} ns | "
        f"Trend amplitude={np.ptp(trend_full):.3f} ns"
    )
    if r2 < 0.1:
        logger.warning(f"[{sat_id}] Low detrend R²={r2:.4f} — trend fit may be poor.")

    result = DetrendResult(
        sat_id=sat_id,
        constellation=constellation,
        orbital_period_hr=orbital_period_hr,
        timestamps=timestamps,
        sise_ns=sise_series,
        trend_poly=trend_poly,
        trend_orbital=trend_orbital,
        trend_solar=trend_solar,
        trend_total=trend_full,
        residual=residual,
        ridge_model=ridge,
        scaler=scaler,
        poly_degree=poly_degree,
        orbital_period_hr_fit=orbital_period_hr,
    )

    if save_to_disk:
        _save_residual(result, output_dir)

    return result


def detrend_all(
    sat_data: dict,
    classifications: dict,
    **kwargs,
) -> dict:
    """
    Detrend all satellites.

    Parameters
    ----------
    sat_data : Dict[sat_id, DataFrame] — from data_loader
    classifications : Dict[sat_id, SatelliteClassification] — from classifier

    Returns
    -------
    Dict[sat_id, DetrendResult]
    """
    results = {}
    for sat_id, df in sat_data.items():
        clf = classifications.get(sat_id)
        constellation = clf.constellation if clf else "UNKNOWN"
        orbital_period_hr = clf.orbital_period_hr if clf else None

        try:
            results[sat_id] = detrend(
                sat_id=sat_id,
                sise_series=df["sise_ns"],
                constellation=constellation,
                orbital_period_hr=orbital_period_hr,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"[{sat_id}] Detrending failed: {e}")

    logger.info(f"Detrended {len(results)}/{len(sat_data)} satellites")
    return results


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _build_feature_matrix(
    t_hours: np.ndarray,
    poly_degree: int,
    orbital_period_hr: float,
    solar_period_hr: float,
) -> np.ndarray:
    """
    Build the design matrix for Ridge regression.

    Columns:
        [1, t, t², t³,  sin(2π·t/P_orb), cos(2π·t/P_orb),
                        sin(2π·t/P_solar), cos(2π·t/P_solar)]

    Parameters
    ----------
    t_hours : np.ndarray
        Time in hours since first observation.
    poly_degree : int
        Degree of polynomial (3 → 4 columns: t⁰, t¹, t², t³).
    orbital_period_hr : float
        Orbital period in hours.
    solar_period_hr : float
        Solar/diurnal period in hours (24.0).

    Returns
    -------
    np.ndarray : shape (n, n_features)
    """
    features = []

    # Polynomial terms
    for deg in range(poly_degree + 1):
        features.append(t_hours ** deg)

    # Orbital harmonic (sin + cos at exact orbital period)
    omega_orb = 2.0 * np.pi / orbital_period_hr
    features.append(np.sin(omega_orb * t_hours))
    features.append(np.cos(omega_orb * t_hours))

    # Solar/diurnal harmonic (sin + cos at 24 hours)
    omega_sol = 2.0 * np.pi / solar_period_hr
    features.append(np.sin(omega_sol * t_hours))
    features.append(np.cos(omega_sol * t_hours))

    # Second orbital harmonic (captures first overtone)
    features.append(np.sin(2 * omega_orb * t_hours))
    features.append(np.cos(2 * omega_orb * t_hours))

    return np.column_stack(features)


def _timestamps_to_hours(
    timestamps: pd.DatetimeIndex,
    t0: pd.Timestamp,
) -> np.ndarray:
    """Convert DatetimeIndex to floating-point hours since t0."""
    return np.array([(t - t0).total_seconds() / 3600.0 for t in timestamps])


def _predict_component_safe(
    ridge: Ridge,
    scaler: StandardScaler,
    X_full: np.ndarray,
    feature_slice: slice,
) -> np.ndarray:
    """Predict contribution of one feature group (zeroing out others)."""
    X_scaled = scaler.transform(X_full)
    coef_subset = np.zeros_like(ridge.coef_)
    coef_subset[feature_slice] = ridge.coef_[feature_slice]
    return X_scaled @ coef_subset


def _save_residual(result: DetrendResult, output_dir: str) -> None:
    """Save detrended residual and trend components to CSV."""
    from pathlib import Path
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df_out = pd.DataFrame({
        "sise_ns": result.sise_ns,
        "trend_total": result.trend_total,
        "trend_poly": result.trend_poly,
        "trend_orbital": result.trend_orbital,
        "trend_solar": result.trend_solar,
        "residual_ns": result.residual,
    }, index=result.timestamps)

    out_file = out_path / f"{result.sat_id}_residual.csv"
    df_out.to_csv(out_file)
    logger.debug(f"[{result.sat_id}] Residual saved to {out_file}")
