"""
evaluate.py — Validation of Day 8 predictions against actual 2026_008.csv.

Usage:
    python src/evaluate.py \
        --predictions outputs/submission.csv \
        --actual-data data/raw/train/2026_008.csv \
        --output      outputs/evaluation_report.csv

What this does
--------------
1.  Processes 2026_008.csv through the same Module 0 pipeline that was used
    for training data → produces per-satellite actual SISE at 15-min cadence.

2.  Loads outputs/submission.csv (96 predictions × N satellites).

3.  Merges predictions and actuals on (satellite_id, timestamp).

4.  Computes the following metrics per satellite AND per evaluation horizon
    (15, 30, 60, 120, 1440 min):

    ┌──────────┬────────────────────────────────────────────────────────────┐
    │ Metric   │ Description                                                │
    ├──────────┼────────────────────────────────────────────────────────────┤
    │ MAE      │ Mean Absolute Error  |actual − predicted| [ns]             │
    │ RMSE     │ Root Mean Squared Error  √mean((actual − pred)²)  [ns]     │
    │ MBE      │ Mean Bias Error  mean(actual − pred)  [ns]  (sign matters) │
    │ σ_ratio  │ mean(|error| / std_pred) — are our uncertainty bands right?│
    │ cov_1σ   │ % of actuals within ±1σ predicted (target: ~68%)          │
    │ cov_2σ   │ % of actuals within ±2σ predicted (target: ~95%)          │
    │ CRPS     │ Continuous Ranked Probability Score (proper scoring rule)  │
    │ err_skew │ Skewness of standardised errors (target: ≈0)              │
    │ err_kurt │ Excess kurtosis of standardised errors (target: <3)       │
    └──────────┴────────────────────────────────────────────────────────────┘

5.  Prints a summary table to stdout and saves full report to CSV.

Notes
-----
- For GPS satellites where 2026_008.csv contains IGS ground truth (Tier A),
  actuals are precise to ~1 ns. For non-GPS (Tier C), actuals are the same
  af0-polynomial proxy used during training — still meaningful because we are
  comparing consistent signal representations.

- If a satellite has no aligned actual values for Day 8 (e.g., satellite was
  unhealthy), it is skipped with a warning.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    EVAL_HORIZONS_MIN,
    PREDICTION_INTERVAL_MIN,
    COMP_CSV_PATTERN,
)
from src.gnss_preprocess import prepare_pipeline_input

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate")


# ── Public API ────────────────────────────────────────────────────────────────

def evaluate_predictions(
    predictions_csv: str | Path,
    actual_data_path: str | Path,
    output_csv: Optional[str | Path] = None,
    actual_cache_dir: Optional[str | Path] = None,
    force_recompute: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare pipeline predictions against Day 8 actual data.

    Parameters
    ----------
    predictions_csv : str | Path
        Path to outputs/submission.csv produced by the pipeline.
    actual_data_path : str | Path
        Path to the 8th-day CSV file (2026_008.csv) OR a directory containing it.
        The file will be processed through Module 0 to compute actual SISE.
    output_csv : str | Path, optional
        Where to save the full per-satellite evaluation report.
        Defaults to outputs/evaluation_report.csv next to predictions_csv.
    actual_cache_dir : str | Path, optional
        Directory to cache the processed Day 8 SISE CSVs (reused on reruns).
        Defaults to a temporary directory.
    force_recompute : bool
        If True, reprocess 2026_008.csv even if cache exists.

    Returns
    -------
    summary_df : pd.DataFrame
        Per-horizon summary metrics (all satellites averaged).
    detail_df : pd.DataFrame
        Per-satellite × per-horizon full metric table.
    """
    predictions_csv = Path(predictions_csv)
    actual_data_path = Path(actual_data_path)

    if output_csv is None:
        output_csv = predictions_csv.parent / "evaluation_report.csv"
    output_csv = Path(output_csv)

    # ── Step 1: Load predictions ──────────────────────────────────────────
    logger.info(f"Loading predictions from {predictions_csv} ...")
    preds = _load_predictions(predictions_csv)
    logger.info(f"  {len(preds)} prediction rows | {preds['satellite_id'].nunique()} satellites")

    # ── Step 2: Process Day 8 through Module 0 ───────────────────────────
    logger.info(f"Processing actual Day 8 data from {actual_data_path} ...")
    if actual_cache_dir is None:
        # Use a persistent temp dir in outputs/ for reuse
        actual_cache_dir = predictions_csv.parent / "day8_actuals_cache"
    actual_cache_dir = Path(actual_cache_dir)

    # Determine data directory and pattern
    if actual_data_path.is_file():
        data_dir = actual_data_path.parent
    else:
        data_dir = actual_data_path

    # Run Module 0 on just the Day 8 file
    actuals_dir = prepare_pipeline_input(
        data_dir=data_dir,
        output_dir=actual_cache_dir,
        n_train_days=8,       # load all 8 files but we only care about day 8
        csv_pattern=_make_day8_pattern(actual_data_path),
        force_recompute=force_recompute,
    )

    actuals = _load_actuals(actuals_dir)
    logger.info(f"  {len(actuals)} actual rows | {actuals['satellite_id'].nunique()} satellites")

    # ── Step 3: Align on common satellites and timestamps ────────────────
    logger.info("Aligning predictions and actuals ...")
    merged = _merge_predictions_actuals(preds, actuals)
    if merged.empty:
        logger.error(
            "No overlapping (satellite_id, timestamp) pairs found. "
            "Check that predictions are for Day 8 and actuals are from 2026_008.csv."
        )
        return pd.DataFrame(), pd.DataFrame()

    n_pairs = len(merged)
    n_sats  = merged["satellite_id"].nunique()
    logger.info(f"  Aligned {n_pairs} prediction-actual pairs across {n_sats} satellites")

    # ── Step 4: Compute metrics ───────────────────────────────────────────
    logger.info("Computing evaluation metrics ...")
    detail_df  = _compute_metrics(merged)
    summary_df = _summarise(detail_df)

    # ── Step 5: Save and print ────────────────────────────────────────────
    detail_df.to_csv(output_csv, index=False)
    logger.info(f"Full evaluation report saved to {output_csv}")

    _print_report(summary_df, detail_df)

    return summary_df, detail_df


# ── Internal helpers ──────────────────────────────────────────────────────────

def _make_day8_pattern(actual_data_path: Path) -> str:
    """
    If actual_data_path points directly to 2026_008.csv, return a pattern
    that matches only that file. Otherwise use the default glob.
    """
    if actual_data_path.is_file():
        return actual_data_path.name   # exact filename
    return COMP_CSV_PATTERN


def _load_predictions(path: Path) -> pd.DataFrame:
    """Load submission.csv and normalise column names."""
    df = pd.read_csv(path)

    # Normalise column names (handle sat_id vs satellite_id)
    df.columns = df.columns.str.strip()
    if "sat_id" in df.columns and "satellite_id" not in df.columns:
        df = df.rename(columns={"sat_id": "satellite_id"})

    # Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    required = {"satellite_id", "timestamp", "mean_ns", "std_ns", "horizon_min"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"submission.csv is missing columns: {missing}. "
            f"Available: {list(df.columns)}"
        )

    return df


def _load_actuals(actuals_dir: Path) -> pd.DataFrame:
    """Load all per-satellite SISE CSVs from the Module 0 output directory."""
    csv_files = sorted(actuals_dir.glob("*_sise.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No *_sise.csv files found in {actuals_dir}. "
            "Module 0 may not have produced output."
        )

    frames = []
    for f in csv_files:
        df = pd.read_csv(f)
        frames.append(df)

    actuals = pd.concat(frames, ignore_index=True)

    # Normalise column names
    if "satellite_id" not in actuals.columns and "sat_id" in actuals.columns:
        actuals = actuals.rename(columns={"sat_id": "satellite_id"})

    actuals["timestamp"] = pd.to_datetime(actuals["timestamp"], utc=True)
    actuals = actuals[["satellite_id", "timestamp", "sise_ns"]].dropna(subset=["sise_ns"])
    return actuals


def _merge_predictions_actuals(
    preds: pd.DataFrame,
    actuals: pd.DataFrame,
) -> pd.DataFrame:
    """
    Inner-join predictions and actuals on (satellite_id, timestamp).
    Timestamps are rounded to the nearest minute before joining to handle
    any sub-minute floating point differences.
    """
    # Round to minute to tolerate minor sub-minute offsets
    preds   = preds.copy()
    actuals = actuals.copy()
    preds["timestamp_key"]   = preds["timestamp"].dt.round("1min")
    actuals["timestamp_key"] = actuals["timestamp"].dt.round("1min")

    merged = pd.merge(
        preds,
        actuals.rename(columns={"sise_ns": "actual_ns"}),
        on=["satellite_id", "timestamp_key"],
        how="inner",
    )

    # Compute standardised error: (actual - predicted) / std
    merged["error_ns"]  = merged["actual_ns"] - merged["mean_ns"]
    merged["z_score"]   = merged["error_ns"] / merged["std_ns"].clip(lower=1e-6)
    merged["abs_error"] = merged["error_ns"].abs()
    merged["sq_error"]  = merged["error_ns"] ** 2
    merged["within_1sigma"] = (merged["z_score"].abs() <= 1.0).astype(int)
    merged["within_2sigma"] = (merged["z_score"].abs() <= 2.0).astype(int)

    return merged


def _crps_gaussian(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """
    Closed-form CRPS for Gaussian predictive distribution.
    CRPS(N(μ,σ²), y) = σ [z(Φ(z) - 0.5) + φ(z) - 1/√π]
    where z = (y - μ) / σ
    Lower is better. Perfect model → 0.
    """
    sigma = np.clip(sigma, 1e-6, None)
    z   = (y - mu) / sigma
    phi = stats.norm.pdf(z)
    Phi = stats.norm.cdf(z)
    crps = sigma * (z * (2 * Phi - 1) + 2 * phi - 1.0 / np.sqrt(np.pi))
    return float(np.mean(crps))


def _compute_metrics(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all metrics per (satellite_id, horizon_label).
    Returns a DataFrame with one row per satellite × horizon combination
    plus an 'ALL' aggregate satellite row.
    """
    rows = []

    # Define horizon bins: each horizon_min value is a checkpoint
    # We also add "all" for the full 24-hr window
    horizon_groups = [("all", None)] + [
        (f"{h}min", h) for h in sorted(EVAL_HORIZONS_MIN)
    ]

    sat_ids = sorted(merged["satellite_id"].unique()) + ["ALL"]

    for sat_id in sat_ids:
        if sat_id == "ALL":
            sat_df = merged
        else:
            sat_df = merged[merged["satellite_id"] == sat_id]

        if sat_df.empty:
            continue

        for horizon_label, horizon_min in horizon_groups:
            if horizon_min is not None:
                # Select only the row(s) at exactly this horizon
                grp = sat_df[sat_df["horizon_min"] == horizon_min]
            else:
                grp = sat_df

            if grp.empty:
                continue

            y     = grp["actual_ns"].values
            mu    = grp["mean_ns"].values
            sigma = grp["std_ns"].values
            err   = grp["error_ns"].values
            z     = grp["z_score"].values
            n     = len(grp)

            mae  = float(np.mean(np.abs(err)))
            rmse = float(np.sqrt(np.mean(err ** 2)))
            mbe  = float(np.mean(err))
            sigma_ratio = float(np.mean(np.abs(err) / np.clip(sigma, 1e-6, None)))
            cov_1sigma  = float(np.mean(np.abs(z) <= 1.0) * 100)
            cov_2sigma  = float(np.mean(np.abs(z) <= 2.0) * 100)
            crps        = _crps_gaussian(y, mu, sigma)
            err_skew    = float(stats.skew(err))       if n >= 4 else np.nan
            err_kurt    = float(stats.kurtosis(err))   if n >= 4 else np.nan
            z_skew      = float(stats.skew(z))         if n >= 4 else np.nan
            z_kurt      = float(stats.kurtosis(z))     if n >= 4 else np.nan

            rows.append({
                "satellite_id":  sat_id,
                "horizon":       horizon_label,
                "n":             n,
                "mae_ns":        round(mae,  3),
                "rmse_ns":       round(rmse, 3),
                "mbe_ns":        round(mbe,  3),
                "sigma_ratio":   round(sigma_ratio, 3),
                "cov_1sigma_%":  round(cov_1sigma, 1),
                "cov_2sigma_%":  round(cov_2sigma, 1),
                "crps":          round(crps, 4),
                "err_skew":      round(err_skew, 3) if not np.isnan(err_skew) else None,
                "err_kurt":      round(err_kurt, 3) if not np.isnan(err_kurt) else None,
                "z_skew":        round(z_skew, 3)   if not np.isnan(z_skew)   else None,
                "z_kurt":        round(z_kurt, 3)   if not np.isnan(z_kurt)   else None,
            })

    return pd.DataFrame(rows)


def _summarise(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Extract the ALL satellite rows for the top-level summary."""
    return detail_df[detail_df["satellite_id"] == "ALL"].copy()


def _print_report(summary_df: pd.DataFrame, detail_df: pd.DataFrame) -> None:
    """Print a clean evaluation summary to stdout."""
    sep = "=" * 78

    print(f"\n{sep}")
    print("EVALUATION REPORT — Day 8 Predictions vs Actuals")
    print(sep)

    # Overall summary
    all_row = summary_df[summary_df["horizon"] == "all"]
    if not all_row.empty:
        r = all_row.iloc[0]
        print(f"\nOVERALL (all satellites, full 24-hr horizon)")
        print(f"  MAE  = {r['mae_ns']:>8.3f} ns")
        print(f"  RMSE = {r['rmse_ns']:>8.3f} ns")
        print(f"  MBE  = {r['mbe_ns']:>8.3f} ns  (bias; ~0 = unbiased)")
        print(f"  CRPS = {r['crps']:>8.4f}       (lower = better)")
        print(f"  σ-ratio  = {r['sigma_ratio']:>6.3f}        (1.0 = perfectly calibrated)")
        print(f"  Cov ±1σ  = {r['cov_1sigma_%']:>5.1f}%       (target ≈ 68%)")
        print(f"  Cov ±2σ  = {r['cov_2sigma_%']:>5.1f}%       (target ≈ 95%)")

    # Per-horizon breakdown
    print(f"\n{'─'*78}")
    print(f"{'Horizon':<12} {'N':>5} {'MAE':>10} {'RMSE':>10} {'MBE':>10} "
          f"{'Cov±1σ':>8} {'Cov±2σ':>8} {'CRPS':>8}")
    print(f"{'─'*78}")
    for _, row in summary_df.iterrows():
        print(
            f"{row['horizon']:<12} {row['n']:>5} "
            f"{row['mae_ns']:>10.3f} {row['rmse_ns']:>10.3f} {row['mbe_ns']:>10.3f} "
            f"{row['cov_1sigma_%']:>7.1f}% {row['cov_2sigma_%']:>7.1f}% "
            f"{row['crps']:>8.4f}"
        )

    # Per-satellite MAE
    sat_rows = detail_df[
        (detail_df["satellite_id"] != "ALL") & (detail_df["horizon"] == "all")
    ].sort_values("mae_ns")

    if not sat_rows.empty:
        print(f"\n{'─'*78}")
        print(f"Per-satellite MAE (full 24h), sorted best → worst:")
        print(f"{'─'*78}")
        print(f"{'Satellite':<12} {'N':>5} {'MAE':>10} {'RMSE':>10} {'MBE':>10} "
              f"{'Cov±1σ':>8} {'Cov±2σ':>8}")
        print(f"{'─'*78}")
        for _, row in sat_rows.iterrows():
            print(
                f"{row['satellite_id']:<12} {row['n']:>5} "
                f"{row['mae_ns']:>10.3f} {row['rmse_ns']:>10.3f} {row['mbe_ns']:>10.3f} "
                f"{row['cov_1sigma_%']:>7.1f}% {row['cov_2sigma_%']:>7.1f}%"
            )

    print(f"\n{sep}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate Day 8 predictions against actual 2026_008.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard evaluation after a pipeline run
  python src/evaluate.py \\
      --predictions outputs/submission.csv \\
      --actual-data data/raw/train/2026_008.csv

  # Save report to a custom path
  python src/evaluate.py \\
      --predictions outputs/submission.csv \\
      --actual-data data/raw/train/2026_008.csv \\
      --output      outputs/day8_eval.csv

  # Force-reprocess the 8th-day CSV (if you changed Module 0)
  python src/evaluate.py \\
      --predictions outputs/submission.csv \\
      --actual-data data/raw/train/2026_008.csv \\
      --force-recompute
        """,
    )
    p.add_argument(
        "--predictions",
        required=True,
        help="Path to outputs/submission.csv from the pipeline run",
    )
    p.add_argument(
        "--actual-data",
        required=True,
        help="Path to 2026_008.csv (the held-out Day 8 file)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output CSV path for full evaluation report (default: same dir as --predictions)",
    )
    p.add_argument(
        "--cache-dir",
        default=None,
        help="Directory to cache processed Day 8 SISE CSVs (default: outputs/day8_actuals_cache)",
    )
    p.add_argument(
        "--force-recompute",
        action="store_true",
        help="Reprocess 2026_008.csv even if cached intermediate CSVs exist",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate_predictions(
        predictions_csv=args.predictions,
        actual_data_path=args.actual_data,
        output_csv=args.output,
        actual_cache_dir=args.cache_dir,
        force_recompute=args.force_recompute,
    )
