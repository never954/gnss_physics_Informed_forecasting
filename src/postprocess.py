"""
postprocess.py — Module 11: Winsorization, output formatting, Gaussianity metrics.

Three responsibilities:

1. WINSORIZATION (Gaussianity safety net)
   Clip all predictions to ±WINSORIZE_CLIP_SIGMA × training_std.
   Prevents extreme outlier predictions from inflating kurtosis.
   Affects < 5% of predictions in normal operation.

2. OUTPUT FORMATTING
   Produce submission-ready CSV rows.

3. GAUSSIANITY EVALUATION
   Compute skewness, excess kurtosis, and Shapiro-Wilk p-value
   across all prediction residuals — these are the competition metrics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    WINSORIZE_CLIP_SIGMA,
    MIN_STD_NS,
    EVAL_HORIZONS_MIN,
    N_PREDICTION_POINTS,
    PREDICTION_INTERVAL_MIN,
    OUTPUT_SUBMISSION,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Winsorization
# ---------------------------------------------------------------------------

def winsorize_predictions(
    mean_pred: np.ndarray,
    std_pred: np.ndarray,
    train_series: pd.Series,
    clip_sigma: float = WINSORIZE_CLIP_SIGMA,
    min_std_ns: float = MIN_STD_NS,
    sat_id: str = "UNKNOWN",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clip prediction means to ±clip_sigma × training_std of the training distribution,
    and enforce a minimum std floor of min_std_ns.

    Why this helps Gaussianity:
    - Any prediction outside ±3σ of what was historically seen is almost certainly wrong
    - Extreme predictions → heavy tails → high kurtosis → bad score
    - Clipping at 3σ: catches outlier predictions while rarely affecting valid ones
      (for Normal distribution, only 0.3% of points lie outside ±3σ)

    Why the std floor:
    - GP models on smooth signals (GLONASS, clean BeiDou) collapse to std ≈ 0
    - This causes 0% coverage even when the mean is accurate
    - MIN_STD_NS = 1.5 ns matches the residual noise floor of atomic clocks

    Parameters
    ----------
    mean_pred : np.ndarray
        Raw model predictions (residual in ns).
    std_pred : np.ndarray
        Prediction uncertainty (std in ns).
    train_series : pd.Series
        Training SISE series (used to compute clip bounds).
    clip_sigma : float
        Clip factor (default 3.0σ).
    min_std_ns : float
        Minimum allowed std (default MIN_STD_NS = 1.5 ns).
    sat_id : str
        For logging.

    Returns
    -------
    mean_clipped : np.ndarray — winsorized mean predictions
    std_floored  : np.ndarray — std with minimum floor applied
    """
    train_valid = train_series.dropna()
    if len(train_valid) == 0:
        logger.warning(f"[{sat_id}] Empty training series for winsorization. Skipping.")
        return mean_pred, np.maximum(std_pred, min_std_ns)

    train_mean = float(train_valid.mean())
    train_std  = float(train_valid.std())

    lo = train_mean - clip_sigma * train_std
    hi = train_mean + clip_sigma * train_std

    n_clipped = int(np.sum((mean_pred < lo) | (mean_pred > hi)))
    if n_clipped > 0:
        frac = n_clipped / len(mean_pred)
        if frac > 0.10:
            logger.warning(
                f"[{sat_id}] Winsorization clipped {n_clipped}/{len(mean_pred)} "
                f"({frac:.1%}) predictions — model may need adjustment."
            )
        else:
            logger.debug(
                f"[{sat_id}] Winsorization clipped {n_clipped}/{len(mean_pred)} "
                f"({frac:.1%}) predictions."
            )

    mean_clipped = np.clip(mean_pred, lo, hi)

    # Apply std floor: no prediction should be more confident than the noise floor
    n_floored = int(np.sum(std_pred < min_std_ns))
    std_floored = np.maximum(std_pred, min_std_ns)
    if n_floored > 0:
        logger.debug(
            f"[{sat_id}] Std floor applied to {n_floored}/{len(std_pred)} points "
            f"(std < {min_std_ns} ns → raised to {min_std_ns} ns)."
        )

    return mean_clipped, std_floored


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_satellite_output(
    sat_id: str,
    future_timestamps: pd.DatetimeIndex,
    mean_ns: np.ndarray,
    std_ns: np.ndarray,
    last_train_timestamp: pd.Timestamp,
) -> pd.DataFrame:
    """
    Format a satellite's prediction into a submission-ready DataFrame.

    Parameters
    ----------
    sat_id : str
    future_timestamps : pd.DatetimeIndex — 96 future timestamps
    mean_ns : np.ndarray — predicted residual mean (ns)
    std_ns  : np.ndarray — predicted residual std (ns)
    last_train_timestamp : pd.Timestamp — last training data point

    Returns
    -------
    pd.DataFrame with columns:
        sat_id | timestamp | mean_ns | std_ns | horizon_min
    """
    horizon_min = np.array([
        int((ts - last_train_timestamp).total_seconds() / 60)
        for ts in future_timestamps
    ])

    df = pd.DataFrame({
        "sat_id":      sat_id,
        "timestamp":   future_timestamps,
        "mean_ns":     mean_ns,
        "std_ns":      std_ns,
        "horizon_min": horizon_min,
    })
    return df


def combine_and_save(
    per_satellite_dfs: Dict[str, pd.DataFrame],
    output_dir: str | Path,
    predictions_dir: str | Path,
) -> pd.DataFrame:
    """
    Combine all per-satellite DataFrames into one submission CSV.
    Also save individual per-satellite CSVs.

    Parameters
    ----------
    per_satellite_dfs : Dict[sat_id, DataFrame]
    output_dir : str | Path — for outputs/submission.csv
    predictions_dir : str | Path — for data/predictions/<sat_id>_prediction.csv

    Returns
    -------
    pd.DataFrame — full submission
    """
    output_dir = Path(output_dir)
    predictions_dir = Path(predictions_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    all_frames = []
    for sat_id, df in sorted(per_satellite_dfs.items()):
        # Save individual satellite prediction
        sat_file = predictions_dir / f"{sat_id}_prediction.csv"
        df.to_csv(sat_file, index=False)
        logger.debug(f"[{sat_id}] Prediction saved to {sat_file}")
        all_frames.append(df)

    if not all_frames:
        raise ValueError("No satellite predictions to save.")

    submission = pd.concat(all_frames, ignore_index=True)
    submission_file = output_dir / "submission.csv"
    submission.to_csv(submission_file, index=False)

    logger.info(
        f"Submission saved: {submission_file} | "
        f"{len(submission)} rows | "
        f"{submission['sat_id'].nunique()} satellites"
    )
    return submission


# ---------------------------------------------------------------------------
# Gaussianity evaluation
# ---------------------------------------------------------------------------

def evaluate_gaussianity(
    submission: pd.DataFrame,
    true_values: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Evaluate Gaussianity of prediction errors (if ground truth available)
    or of predictions themselves (diagnostic mode).

    Parameters
    ----------
    submission : pd.DataFrame — model predictions (from combine_and_save)
    true_values : pd.DataFrame, optional — if provided, compute residuals first

    Returns
    -------
    pd.DataFrame with Gaussianity metrics per satellite and horizon
    """
    rows = []

    if true_values is not None:
        # Compute prediction errors
        merged = submission.merge(
            true_values, on=["sat_id", "timestamp"], suffixes=("_pred", "_true")
        )
        errors_col = "error_ns"
        merged[errors_col] = merged["mean_ns"] - merged["true_ns"]
    else:
        # Diagnostic: evaluate gaussianity of predictions themselves
        merged = submission.copy()
        errors_col = "mean_ns"

    for sat_id, sat_df in merged.groupby("sat_id"):
        values = sat_df[errors_col].dropna().values

        if len(values) < 8:
            continue

        skewness  = float(stats.skew(values))
        kurtosis  = float(stats.kurtosis(values))  # excess kurtosis (normal=0)
        sw_stat, sw_p = (None, None)
        if len(values) <= 5000:
            sw_stat, sw_p = stats.shapiro(values)

        rows.append({
            "sat_id":           sat_id,
            "horizon":          "all",
            "n_points":         len(values),
            "mean":             float(np.mean(values)),
            "std":              float(np.std(values)),
            "skewness":         skewness,
            "excess_kurtosis":  kurtosis,
            "shapiro_p":        float(sw_p) if sw_p else None,
            "is_gaussian":      (abs(skewness) < 1.0) and (abs(kurtosis) < 3.0),
        })

    # Per-horizon breakdown
    for horizon in EVAL_HORIZONS_MIN:
        subset = merged[merged["horizon_min"] <= horizon] if "horizon_min" in merged.columns else merged
        values = subset[errors_col].dropna().values
        if len(values) < 8:
            continue

        skewness = float(stats.skew(values))
        kurtosis = float(stats.kurtosis(values))
        rows.append({
            "sat_id":           "ALL",
            "horizon":          f"{horizon}min",
            "n_points":         len(values),
            "mean":             float(np.mean(values)),
            "std":              float(np.std(values)),
            "skewness":         skewness,
            "excess_kurtosis":  kurtosis,
            "shapiro_p":        None,
            "is_gaussian":      (abs(skewness) < 1.0) and (abs(kurtosis) < 3.0),
        })

    result = pd.DataFrame(rows)

    # Log summary
    if len(result) > 0:
        all_row = result[result["sat_id"] == "ALL"]
        if len(all_row) > 0:
            for _, row in all_row.iterrows():
                logger.info(
                    f"Gaussianity [{row['horizon']}]: "
                    f"skew={row['skewness']:.3f} | "
                    f"kurt={row['excess_kurtosis']:.3f} | "
                    f"gaussian={'✓' if row['is_gaussian'] else '✗'}"
                )

    return result


def print_gaussianity_report(metrics: pd.DataFrame) -> None:
    """Pretty-print Gaussianity evaluation to console."""
    print("\n" + "=" * 70)
    print("GAUSSIANITY EVALUATION REPORT")
    print("=" * 70)
    print(f"{'Satellite':<12} {'Horizon':<10} {'N':<6} {'Skew':<8} {'Kurt':<8} {'Gaussian'}")
    print("-" * 70)
    for _, row in metrics.iterrows():
        flag = "✓" if row["is_gaussian"] else "✗"
        print(
            f"{row['sat_id']:<12} {str(row['horizon']):<10} {row['n_points']:<6} "
            f"{row['skewness']:<8.3f} {row['excess_kurtosis']:<8.3f} {flag}"
        )
    print("=" * 70)
    print("Target: |skew| < 1.0 AND |kurtosis| < 3.0")
    print()
