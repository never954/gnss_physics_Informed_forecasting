"""
data_loader.py — Module 2: Ingest and validate GNSS error CSV data.

Responsibilities:
  - Resolve column name aliases (competition-day flexible)
  - Split multi-satellite CSVs into per-satellite DataFrames
  - Validate 15-min cadence, fill small gaps, flag large gaps
  - Return Dict[sat_id → DataFrame] ready for downstream modules

Expected input: CSV with columns (names configurable in config.py):
    timestamp | sat_id | clock_error_ns | eph_error_m
    -OR-
    timestamp | sat_id | sise_ns
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    COL_TIMESTAMP, COL_SAT_ID, COL_CLOCK_ERR, COL_EPH_ERR, COL_SISE,
    COL_TIMESTAMP_ALIASES, COL_SAT_ID_ALIASES,
    COL_CLOCK_ERR_ALIASES, COL_EPH_ERR_ALIASES, COL_SISE_ALIASES,
    PREDICTION_INTERVAL_MIN,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_satellite_data(
    data_path: str | Path,
    expected_days: int = 7,
    fill_small_gaps: bool = True,
    max_gap_to_fill: int = 2,
) -> Dict[str, pd.DataFrame]:
    """
    Load all satellite error data from a directory or single CSV file.

    Parameters
    ----------
    data_path : str | Path
        Path to a CSV file or a directory of CSV files.
    expected_days : int
        Expected number of training days (default 7).
        Used to warn if a satellite has significantly fewer points.
    fill_small_gaps : bool
        If True, linearly interpolate gaps of ≤ max_gap_to_fill steps.
    max_gap_to_fill : int
        Max consecutive missing steps to fill automatically.

    Returns
    -------
    Dict[sat_id, pd.DataFrame]
        Keys: satellite PRN strings (e.g., "G01", "E05").
        Values: DataFrames indexed by pd.DatetimeIndex with columns:
            - clock_error_ns  (float, may be NaN if only SISE provided)
            - eph_error_m     (float, may be NaN if only SISE provided)
            - sise_ns         (float, clock_error_ns + scaled eph contribution)
    """
    data_path = Path(data_path)

    if data_path.is_file():
        csv_files = [data_path]
    elif data_path.is_dir():
        csv_files = sorted(data_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_path}")
    else:
        raise FileNotFoundError(f"Path does not exist: {data_path}")

    logger.info(f"Loading {len(csv_files)} CSV file(s) from {data_path}")

    # Load and concatenate all files
    frames: List[pd.DataFrame] = []
    for f in csv_files:
        df = _load_single_csv(f)
        frames.append(df)
        logger.debug(f"  Loaded {f.name}: {len(df)} rows")

    combined = pd.concat(frames, ignore_index=True)

    # Split by satellite
    sat_col = _resolve_column(combined, COL_SAT_ID_ALIASES, "sat_id")
    sat_ids = combined[sat_col].unique()
    logger.info(f"Found {len(sat_ids)} satellites: {sorted(sat_ids)}")

    result: Dict[str, pd.DataFrame] = {}
    for sat_id in sorted(sat_ids):
        sat_df = combined[combined[sat_col] == sat_id].copy()
        sat_df = _validate_and_clean(
            sat_df, sat_id, expected_days, fill_small_gaps, max_gap_to_fill
        )
        if sat_df is not None:
            result[sat_id] = sat_df

    logger.info(f"Successfully loaded {len(result)} satellites")
    return result


def get_train_test_split(
    sat_data: Dict[str, pd.DataFrame],
    n_train_days: int = 7,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.Series]]:
    """
    Split loaded data into training portion and the last-known timestamp info.

    For competition use: all data is training; this returns full data plus
    the future timestamps for prediction (Day 8, 96 points).

    Returns
    -------
    train_data : Dict[sat_id, DataFrame]
        The 7-day training DataFrames.
    future_timestamps : Dict[sat_id, DatetimeIndex]
        96-step future timestamps for Day 8 predictions.
    """
    train_data = {}
    future_timestamps = {}

    points_per_day = 24 * 60 // PREDICTION_INTERVAL_MIN  # 96

    for sat_id, df in sat_data.items():
        # Use all available data as training
        train_data[sat_id] = df

        # Build future timestamps starting 15 min after last training point
        last_ts = df.index[-1]
        freq = pd.Timedelta(minutes=PREDICTION_INTERVAL_MIN)
        future_ts = pd.date_range(
            start=last_ts + freq,
            periods=96,
            freq=freq,
        )
        future_timestamps[sat_id] = future_ts

    return train_data, future_timestamps


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_single_csv(path: Path) -> pd.DataFrame:
    """Load one CSV, trying multiple encodings if needed."""
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(path, encoding=encoding)
            logger.debug(f"  Loaded {path.name} with encoding={encoding}")
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path} with any tried encoding")


def _resolve_column(df: pd.DataFrame, aliases: List[str], canonical_name: str) -> str:
    """
    Find the first alias that exists as a column name (case-insensitive).
    Returns the actual column name in the DataFrame.
    Raises ValueError if none found.
    """
    df_cols_lower = {c.lower(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in df_cols_lower:
            found = df_cols_lower[alias.lower()]
            if found != canonical_name:
                logger.debug(f"  Column alias: '{found}' → treated as '{canonical_name}'")
            return found
    raise ValueError(
        f"Could not find column '{canonical_name}'. "
        f"Tried aliases: {aliases}. "
        f"Available columns: {list(df.columns)}. "
        f"Edit COL_{canonical_name.upper()}_ALIASES in src/config.py."
    )


def _parse_timestamps(ts_series: pd.Series) -> pd.DatetimeIndex:
    """Parse timestamp column to DatetimeIndex, trying multiple formats."""
    # Try numeric (Unix epoch in seconds)
    if pd.api.types.is_numeric_dtype(ts_series):
        return pd.to_datetime(ts_series, unit="s", utc=True)

    # Try ISO 8601 and other string formats
    try:
        return pd.to_datetime(ts_series, utc=True)
    except Exception:
        # Try GPS week + seconds format (e.g., "2284 432000.0")
        logger.warning("Could not parse timestamps directly; trying GPS week/second format")
        raise ValueError(
            "Cannot parse timestamp column. Ensure it is ISO 8601, Unix epoch (seconds), "
            "or a pandas-recognized format. See DATA_ROUTING.md."
        )


def _validate_and_clean(
    df: pd.DataFrame,
    sat_id: str,
    expected_days: int,
    fill_small_gaps: bool,
    max_gap_to_fill: int,
) -> Optional[pd.DataFrame]:
    """
    Validate one satellite's DataFrame, clean it, and return a standardized form.
    Returns None if the satellite has too few points to be useful.
    """
    # --- Resolve timestamp column ---
    ts_col = _resolve_column(df, COL_TIMESTAMP_ALIASES, "timestamp")
    df.index = _parse_timestamps(df[ts_col])
    df.index.name = "timestamp"
    df = df.sort_index()

    # --- Resolve error columns ---
    # Try SISE directly, else compute from clock + ephemeris
    sise_available = False
    try:
        sise_col = _resolve_column(df, COL_SISE_ALIASES, "sise_ns")
        df["sise_ns"] = pd.to_numeric(df[sise_col], errors="coerce")
        sise_available = True
    except ValueError:
        pass

    clock_available = False
    try:
        clk_col = _resolve_column(df, COL_CLOCK_ERR_ALIASES, "clock_error_ns")
        df["clock_error_ns"] = pd.to_numeric(df[clk_col], errors="coerce")
        clock_available = True
    except ValueError:
        pass

    eph_available = False
    try:
        eph_col = _resolve_column(df, COL_EPH_ERR_ALIASES, "eph_error_m")
        df["eph_error_m"] = pd.to_numeric(df[eph_col], errors="coerce")
        eph_available = True
    except ValueError:
        pass

    # Build sise_ns if not directly provided
    if not sise_available:
        if clock_available:
            # SISE ≈ clock_error_ns (ephemeris contribution is typically < 1 ns equivalent)
            # Full SISE = sqrt((clk_err * c)^2 + radial_err^2) / c in ns
            # Simplified: use clock error as primary signal
            df["sise_ns"] = df["clock_error_ns"].copy()
            if eph_available:
                # Add radial ephemeris contribution (convert m → ns: divide by c in m/ns)
                c_m_per_ns = 0.2998  # speed of light in m/ns
                df["sise_ns"] = df["sise_ns"] + df["eph_error_m"] / c_m_per_ns
        else:
            logger.error(
                f"[{sat_id}] No usable error columns found. "
                f"Need '{COL_SISE}' or '{COL_CLOCK_ERR}'. Skipping satellite."
            )
            return None

    if not clock_available:
        df["clock_error_ns"] = np.nan
    if not eph_available:
        df["eph_error_m"] = np.nan

    # --- Select and rename to canonical columns ---
    out = df[["clock_error_ns", "eph_error_m", "sise_ns"]].copy()

    # --- Check for minimum data ---
    min_points = max(48, expected_days * 96 // 10)  # At least 10% of expected data
    if len(out) < min_points:
        logger.warning(
            f"[{sat_id}] Only {len(out)} points (expected ~{expected_days * 96}). "
            f"Below minimum threshold of {min_points}. Skipping."
        )
        return None

    # --- Check and handle timestamp spacing ---
    freq = pd.Timedelta(minutes=PREDICTION_INTERVAL_MIN)
    expected_index = pd.date_range(out.index[0], out.index[-1], freq=freq)
    missing_ts = expected_index.difference(out.index)

    if len(missing_ts) > 0:
        frac_missing = len(missing_ts) / len(expected_index)
        logger.warning(f"[{sat_id}] {len(missing_ts)} missing timestamps ({frac_missing:.1%})")

        if frac_missing > 0.20:
            logger.error(f"[{sat_id}] Too many gaps (>20%). Skipping satellite.")
            return None

        # Reindex to full cadence (introduces NaNs at gaps)
        out = out.reindex(expected_index)

    # --- Fill small gaps via linear interpolation ---
    if fill_small_gaps:
        # Only fill runs of NaNs up to max_gap_to_fill consecutive
        out = _interpolate_small_gaps(out, max_gap_to_fill)

    # --- Final NaN check ---
    nan_frac = out["sise_ns"].isna().mean()
    if nan_frac > 0.10:
        logger.warning(f"[{sat_id}] {nan_frac:.1%} NaN in sise_ns after interpolation.")

    logger.info(
        f"[{sat_id}] Loaded {len(out)} points | "
        f"NaN: {nan_frac:.1%} | "
        f"Range: [{out['sise_ns'].min():.2f}, {out['sise_ns'].max():.2f}] ns"
    )
    return out


def _interpolate_small_gaps(df: pd.DataFrame, max_gap: int) -> pd.DataFrame:
    """
    Linearly interpolate NaN runs of length ≤ max_gap.
    Leaves longer NaN runs untouched (they stay NaN).
    """
    df = df.copy()
    for col in ["sise_ns", "clock_error_ns", "eph_error_m"]:
        if col not in df.columns:
            continue
        s = df[col]
        # Mark which NaNs are in small runs
        is_nan = s.isna()
        if not is_nan.any():
            continue
        # Find run lengths
        runs = (is_nan != is_nan.shift()).cumsum()
        run_sizes = is_nan.groupby(runs).transform("sum")
        small_gap_mask = is_nan & (run_sizes <= max_gap)

        # Only interpolate at small-gap positions
        interpolated = s.interpolate(method="time")
        df.loc[small_gap_mask, col] = interpolated[small_gap_mask]

    return df


# ---------------------------------------------------------------------------
# Diagnostic utility
# ---------------------------------------------------------------------------

def describe_dataset(sat_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Return a summary DataFrame describing all loaded satellites.
    Useful for quick sanity checks.
    """
    rows = []
    for sat_id, df in sat_data.items():
        rows.append({
            "sat_id": sat_id,
            "n_points": len(df),
            "start": df.index[0],
            "end": df.index[-1],
            "n_nan": df["sise_ns"].isna().sum(),
            "mean_ns": df["sise_ns"].mean(),
            "std_ns": df["sise_ns"].std(),
            "min_ns": df["sise_ns"].min(),
            "max_ns": df["sise_ns"].max(),
        })
    return pd.DataFrame(rows).set_index("sat_id")
