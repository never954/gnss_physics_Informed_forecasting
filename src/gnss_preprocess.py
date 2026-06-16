"""
gnss_preprocess.py — Module 0: Competition Data Adapter.

Converts the 78-column competition CSV format (2026_001.csv … 2026_007.csv)
into per-satellite SISE time series at 15-min cadence, ready for Module 2
(data_loader.py) to consume without any changes.

── What this module does ─────────────────────────────────────────────────────
1. Load all day-of-year CSVs (2026_001 … 2026_007) → one big DataFrame
2. Split by satellite_id
3. For each satellite, compute SISE (or a physically meaningful proxy):

   ┌────────────────────┬──────────────────────────────────────────────────┐
   │ Tier A             │ GPS satellite with IGS ground truth at this epoch │
   │                    │  → SISE = (broadcast_clock - igs_clock) × 1e9 ns │
   │                    │         + radial_ephemeris_error / c  [ns]        │
   ├────────────────────┼──────────────────────────────────────────────────┤
   │ Tier B             │ GPS satellite, no IGS alignment at this epoch     │
   │                    │  → sise_proxy = eval_clock_poly(af0,af1,af2) × 1e9│
   ├────────────────────┼──────────────────────────────────────────────────┤
   │ Tier C             │ Non-GPS (Galileo, GLONASS, BeiDou, QZSS, …)      │
   │                    │  → sise_proxy = eval_clock_poly(af0,af1,af2) × 1e9│
   └────────────────────┴──────────────────────────────────────────────────┘

4. Resample irregular broadcast cadence → 15-min grid (zero-order hold)
5. Save per-satellite CSVs  →  data_loader.py picks them up unchanged

── Why zero-order hold for resampling? ──────────────────────────────────────
Broadcast ephemeris is valid until the NEXT upload. A receiver always uses
the most recently received message. Holding the last value until the next
broadcast is therefore physically correct — it mirrors how receivers behave.
We cap the hold at BROADCAST_MAX_FILL_STEPS (16 × 15min = 4 hours). Beyond
that, the data is too stale and is left as NaN.

── Reference ──────────────────────────────────────────────────────────────────
Keplerian→ECEF: IS-GPS-200L §20.3.3.4.3.1 (Table 20-IV algorithm)
Clock polynomial: IS-GPS-200L §20.3.3.3.3.1
Time systems:
  GPS   epoch = 1980-01-06 00:00:00 UTC
  GAL   epoch = 1999-08-22 00:00:00 UTC
  BDS   epoch = 2006-01-01 00:00:00 UTC
  NavIC epoch = 1999-08-22 00:00:00 UTC
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    # Competition column constants (Section 11)
    COMP_COL_SAT_ID, COMP_COL_EPOCH, COMP_COL_CONST,
    COMP_COL_AF0, COMP_COL_AF1, COMP_COL_AF2,
    COMP_COL_TOC_GPS, COMP_COL_GPS_WEEK,
    COMP_COL_TOC_GAL, COMP_COL_GAL_WEEK,
    COMP_COL_TOC_BDS, COMP_COL_BDS_WEEK,
    COMP_COL_TOC_IRN, COMP_COL_IRN_WEEK,
    COMP_COL_IGS_CLK, COMP_COL_IGS_X, COMP_COL_IGS_Y, COMP_COL_IGS_Z,
    COMP_KEPLERIAN_COLS,
    BROADCAST_MAX_FILL_STEPS,
    COMP_CSV_PATTERN,
    PREDICTION_INTERVAL_MIN,
    N_PREDICTION_POINTS,
    C_M_PER_NS,
    GPS_USE_IGS,
)

logger = logging.getLogger(__name__)

# ── Physical constants ────────────────────────────────────────────────────────
GM_WGS84: float = 3.986005e14        # m³/s²  (WGS-84 gravitational constant)
OMEGA_E:  float = 7.2921151467e-5    # rad/s  (Earth rotation rate, WGS-84)
GPS_SEC_PER_WEEK: float = 604800.0   # seconds per GPS week

# GNSS system epochs (as UTC timestamps)
_GPS_EPOCH = pd.Timestamp("1980-01-06 00:00:00", tz="UTC")
_GAL_EPOCH = pd.Timestamp("1999-08-22 00:00:00", tz="UTC")
_BDS_EPOCH = pd.Timestamp("2006-01-01 00:00:00", tz="UTC")
_IRN_EPOCH = pd.Timestamp("1999-08-22 00:00:00", tz="UTC")  # same as Galileo

# Which week/toc columns to use per constellation
_CONST_TIME_CONFIG: Dict[str, Tuple[str, str, pd.Timestamp]] = {
    "GPS":     (COMP_COL_TOC_GPS, COMP_COL_GPS_WEEK, _GPS_EPOCH),
    "QZSS":    (COMP_COL_TOC_GPS, COMP_COL_GPS_WEEK, _GPS_EPOCH),
    "Galileo": (COMP_COL_TOC_GAL, COMP_COL_GAL_WEEK, _GAL_EPOCH),
    "BeiDou":  (COMP_COL_TOC_BDS, COMP_COL_BDS_WEEK, _BDS_EPOCH),
    "NavIC":   (COMP_COL_TOC_IRN, COMP_COL_IRN_WEEK, _IRN_EPOCH),
    # GLONASS uses Moscow time; we fall back to direct af0 at each epoch
    "GLONASS": (None, None, None),
    "SBAS":    (None, None, None),
}


# ── Public API ────────────────────────────────────────────────────────────────

def prepare_pipeline_input(
    data_dir: str | Path,
    output_dir: str | Path,
    n_train_days: int = 7,
    csv_pattern: str = COMP_CSV_PATTERN,
    max_fill_steps: int = BROADCAST_MAX_FILL_STEPS,
    force_recompute: bool = False,
    gps_use_igs: bool = GPS_USE_IGS,
) -> Path:
    """
    Convert competition 78-column CSVs into per-satellite SISE CSVs
    that data_loader.py can consume directly.

    Parameters
    ----------
    data_dir : str | Path
        Directory containing 2026_001.csv … 2026_007.csv.
    output_dir : str | Path
        Where to write <sat_id>_sise.csv files (one per satellite).
        data_loader.py should be pointed at this directory.
    n_train_days : int
        Number of day-of-year files to load (default 7, i.e. 2026_001–007).
    csv_pattern : str
        Glob pattern for competition CSVs (default "2026_*.csv").
    max_fill_steps : int
        Maximum number of 15-min steps to forward-fill a stale broadcast epoch.
    force_recompute : bool
        If False, skip satellites whose output CSV already exists.
    gps_use_igs : bool
        If True, use IGS-subtracted SISE for GPS Tier A records (accurate but sparse).
        If False (default), always use the af0 polynomial proxy for GPS, ensuring
        consistent signal definition between training and evaluation. Set to False
        unless you have a specific reason to use IGS ground truth.

    Returns
    -------
    Path
        The output_dir path (pass this to data_loader.load_satellite_data()).
    """
    data_dir  = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: load all CSVs ─────────────────────────────────────────────
    logger.info(f"[Module 0] Loading competition CSVs from {data_dir} ...")
    raw = _load_competition_csvs(data_dir, csv_pattern, n_train_days)
    logger.info(f"[Module 0] Loaded {len(raw):,} broadcast records "
                f"from {raw[COMP_COL_EPOCH].nunique()} unique epochs")

    # ── Step 2: parse epoch column ────────────────────────────────────────
    raw[COMP_COL_EPOCH] = pd.to_datetime(raw[COMP_COL_EPOCH], utc=True)

    # ── Step 3: split by satellite & compute SISE ─────────────────────────
    sat_groups = raw.groupby(COMP_COL_SAT_ID)
    n_sats = len(sat_groups)
    logger.info(f"[Module 0] Processing {n_sats} satellites ...")

    # Build the 15-min grid for the full training span
    grid_start = raw[COMP_COL_EPOCH].min().floor("15min")
    grid_end   = raw[COMP_COL_EPOCH].max().ceil("15min")
    full_grid  = pd.date_range(grid_start, grid_end,
                               freq=f"{PREDICTION_INTERVAL_MIN}min", tz="UTC")

    processed = 0
    skipped   = 0
    for sat_id, sat_df in sat_groups:
        out_path = output_dir / f"{sat_id}_sise.csv"
        if out_path.exists() and not force_recompute:
            logger.debug(f"[{sat_id}] Already exists, skipping.")
            skipped += 1
            continue

        constellation = sat_df[COMP_COL_CONST].iloc[0]

        try:
            sise_series = _compute_sise(sat_id, sat_df, constellation,
                                        gps_use_igs=gps_use_igs)
            resampled   = _resample_to_15min(sat_id, sise_series, full_grid, max_fill_steps)
            _save_intermediate_csv(sat_id, resampled, out_path)
            processed += 1
        except Exception as exc:
            logger.error(f"[{sat_id}] Failed: {exc}. Skipping satellite.")

    logger.info(
        f"[Module 0] Done. Processed: {processed} | Skipped (cached): {skipped} "
        f"| Failed: {n_sats - processed - skipped}"
    )
    return output_dir


def get_prediction_timestamps(
    data_dir: str | Path,
    csv_pattern: str = COMP_CSV_PATTERN,
) -> pd.DatetimeIndex:
    """
    Build the 96-step Day 8 prediction grid from the latest epoch in training data.

    Returns
    -------
    pd.DatetimeIndex  (96 points, 15-min cadence, starting 15min after Day 7 end)
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob(csv_pattern))
    if not files:
        raise FileNotFoundError(f"No competition CSVs found in {data_dir}")

    # Read only the epoch column from the last training file for speed
    last_file = files[-1]
    epochs = pd.read_csv(last_file, usecols=[COMP_COL_EPOCH])[COMP_COL_EPOCH]
    last_epoch = pd.to_datetime(epochs, utc=True).max()

    freq   = pd.Timedelta(minutes=PREDICTION_INTERVAL_MIN)
    return pd.date_range(
        start=last_epoch + freq,
        periods=N_PREDICTION_POINTS,
        freq=freq,
        tz="UTC",
    )


# ── SISE computation per satellite ───────────────────────────────────────────

def _compute_sise(
    sat_id: str,
    sat_df: pd.DataFrame,
    constellation: str,
    gps_use_igs: bool = GPS_USE_IGS,
) -> pd.Series:
    """
    Compute a per-epoch SISE (or proxy) Series indexed by epoch timestamp.

    Signal definition
    -----------------
    gps_use_igs=False (default, recommended):
        ALL constellations use the af0 clock polynomial proxy exclusively.
        Every broadcast epoch contributes a value → dense, consistent signal.
        Training and evaluation use the IDENTICAL definition → MBE ≈ 0.

    gps_use_igs=True:
        GPS/QZSS Tier A records (those with IGS ground truth in the same row)
        use the IGS-subtracted clock error + Keplerian radial ephemeris error.
        All other records still use the polynomial proxy.
        WARNING: mixing Tier A (~0-50 ns) with Tier B (~300k ns) in the same
        time series creates artificial discontinuities and inflates GPS errors.
        Only use this flag if the evaluation script also uses IGS truth.

    Returns
    -------
    pd.Series  index=epoch (UTC), values=sise_ns (float)
    """
    sat_df = sat_df.sort_values(COMP_COL_EPOCH).copy()
    epochs = sat_df[COMP_COL_EPOCH]

    # Evaluate broadcast clock polynomial for ALL rows (always needed)
    clock_poly_s = _eval_clock_poly_series(sat_df, constellation)

    # Default: proxy for all rows (covers all constellations + GPS when gps_use_igs=False)
    clock_ns  = clock_poly_s.values * 1e9
    eph_err_m = np.full(len(sat_df), np.nan)

    # Tier A: GPS/QZSS with IGS ground truth — only when explicitly enabled
    if gps_use_igs and constellation in ("GPS", "QZSS"):
        has_igs = (
            sat_df[COMP_COL_IGS_CLK].notna()
            & sat_df[COMP_COL_IGS_X].notna()
        )
        igs_idx = has_igs[has_igs].index

        if len(igs_idx) > 0:
            # Overwrite polynomial proxy with IGS-subtracted clock error
            for row_idx in igs_idx:
                arr_i = sat_df.index.get_loc(row_idx)
                bcast_clk = float(clock_poly_s.loc[row_idx])
                igs_clk   = float(sat_df.loc[row_idx, COMP_COL_IGS_CLK])
                clock_ns[arr_i] = (bcast_clk - igs_clk) * 1e9

            # Radial ephemeris error via Keplerian→ECEF (GPS only)
            if all(c in sat_df.columns for c in COMP_KEPLERIAN_COLS):
                kep_ok = sat_df.loc[igs_idx, COMP_KEPLERIAN_COLS].notna().all(axis=1)
                for row_idx in igs_idx:
                    if not kep_ok.loc[row_idx]:
                        continue
                    row = sat_df.loc[row_idx]
                    try:
                        t_gps     = _epoch_to_gps_sow(row[COMP_COL_EPOCH])
                        bcast_xyz = _keplerian_to_ecef(row, t_gps)
                        igs_xyz   = np.array([
                            row[COMP_COL_IGS_X] * 1000,
                            row[COMP_COL_IGS_Y] * 1000,
                            row[COMP_COL_IGS_Z] * 1000,
                        ])
                        r_hat = igs_xyz / (np.linalg.norm(igs_xyz) + 1e-12)
                        arr_i = sat_df.index.get_loc(row_idx)
                        eph_err_m[arr_i] = float(np.dot(bcast_xyz - igs_xyz, r_hat))
                    except Exception:
                        pass

    # Combine clock + ephemeris into SISE
    sise_ns = clock_ns.copy()
    valid_eph = ~np.isnan(eph_err_m)
    sise_ns[valid_eph] += eph_err_m[valid_eph] / C_M_PER_NS

    n_valid = int(np.sum(~np.isnan(sise_ns)))
    if gps_use_igs and constellation in ("GPS", "QZSS"):
        has_igs = sat_df[COMP_COL_IGS_CLK].notna() & sat_df[COMP_COL_IGS_X].notna()
        logger.info(
            f"[{sat_id}] {constellation} | {len(sat_df)} epochs | "
            f"Tier A (IGS): {has_igs.sum()} | proxy: {(~has_igs).sum()} | "
            f"valid SISE: {n_valid}"
        )
    else:
        logger.info(
            f"[{sat_id}] {constellation} | {len(sat_df)} epochs | "
            f"proxy-only (GPS_USE_IGS={gps_use_igs}) | valid SISE: {n_valid}"
        )

    return pd.Series(sise_ns, index=epochs.values, name="sise_ns")


# ── Clock polynomial evaluation ───────────────────────────────────────────────

def _eval_clock_poly_series(
    sat_df: pd.DataFrame,
    constellation: str,
) -> pd.Series:
    """
    Evaluate af0 + af1*(t - t_oc) + af2*(t - t_oc)² for every row.

    Returns Series (same index as sat_df) with clock correction in SECONDS.
    """
    af0 = pd.to_numeric(sat_df[COMP_COL_AF0], errors="coerce").values
    af1 = pd.to_numeric(sat_df[COMP_COL_AF1], errors="coerce").fillna(0).values
    af2 = pd.to_numeric(sat_df[COMP_COL_AF2], errors="coerce").fillna(0).values

    dt = _compute_dt_series(sat_df, constellation)

    clock_s = af0 + af1 * dt + af2 * dt ** 2
    return pd.Series(clock_s, index=sat_df.index)


def _compute_dt_series(
    sat_df: pd.DataFrame,
    constellation: str,
) -> np.ndarray:
    """
    Compute dt = t_epoch - t_oc for each row (in seconds).

    Strategy per constellation:
    - GPS/QZSS: convert epoch to GPS SoW; dt = gps_sow - toe_sec_gps_week
    - Galileo:  same logic with Galileo week
    - BeiDou:   same logic with BDS week
    - NavIC:    same logic with NavIC week
    - GLONASS/SBAS: no polynomial reference time available → dt = 0 (use af0 directly)
    """
    config = _CONST_TIME_CONFIG.get(constellation, (None, None, None))
    toc_col, week_col, epoch_ref = config

    epochs = sat_df[COMP_COL_EPOCH]

    if toc_col is None or toc_col not in sat_df.columns:
        # GLONASS / SBAS / unknown: use af0 directly (dt = 0)
        return np.zeros(len(sat_df))

    toc_sow  = pd.to_numeric(sat_df[toc_col],  errors="coerce").values
    week_num = pd.to_numeric(sat_df[week_col],  errors="coerce").values

    # Convert epoch timestamp → seconds since constellation epoch
    epoch_secs = np.array([
        (t - epoch_ref).total_seconds()
        for t in pd.DatetimeIndex(epochs)
    ])

    # Epoch as SoW within the constellation week
    epoch_sow = epoch_secs % GPS_SEC_PER_WEEK

    dt = epoch_sow - toc_sow

    # Handle week boundary crossovers
    dt = np.where(dt >  302400, dt - GPS_SEC_PER_WEEK, dt)
    dt = np.where(dt < -302400, dt + GPS_SEC_PER_WEEK, dt)

    return dt


# ── Keplerian → ECEF (IS-GPS-200L §20.3.3.4.3.1) ─────────────────────────────

def _keplerian_to_ecef(row: pd.Series, t_gps_sow: float) -> np.ndarray:
    """
    Compute satellite ECEF position [x, y, z] in metres from Keplerian elements.

    Parameters
    ----------
    row : pd.Series  — one row from the broadcast ephemeris DataFrame
    t_gps_sow : float — epoch expressed as GPS seconds-of-week

    Returns
    -------
    np.ndarray shape (3,) — [x_m, y_m, z_m] in ECEF
    """
    sqrt_a   = float(row["sqrt_a_sqrt_m"])
    e        = float(row["e_eccentricity"])
    i0       = float(row["i0_rad"])
    OMEGA0   = float(row["omega0_rad"])
    omega    = float(row["omega_rad"])
    M0       = float(row["m0_rad"])
    delta_n  = float(row["delta_n_rad_sec"])
    idot     = float(row["idot_rad_sec"])
    OMEGA_dot = float(row["omega_dot_rad_sec"])
    Crs      = float(row["crs_m"])
    Crc      = float(row["crc_m"])
    Cus      = float(row["cus_rad"])
    Cuc      = float(row["cuc_rad"])
    Cis      = float(row["cis_rad"])
    Cic      = float(row["cic_rad"])
    toe      = float(row["toe_sec_gps_week"])

    a = sqrt_a ** 2

    # Time from reference epoch (handle week boundary)
    tk = t_gps_sow - toe
    if tk >  302400: tk -= GPS_SEC_PER_WEEK
    elif tk < -302400: tk += GPS_SEC_PER_WEEK

    # Corrected mean motion
    n0 = np.sqrt(GM_WGS84 / a ** 3)
    n  = n0 + delta_n

    # Mean anomaly
    Mk = M0 + n * tk

    # Eccentric anomaly (Kepler's equation — 10 fixed-point iterations)
    Ek = Mk
    for _ in range(10):
        Ek = Mk + e * np.sin(Ek)

    # True anomaly
    sin_nu = np.sqrt(1.0 - e ** 2) * np.sin(Ek) / (1.0 - e * np.cos(Ek))
    cos_nu = (np.cos(Ek) - e) / (1.0 - e * np.cos(Ek))
    nu_k   = np.arctan2(sin_nu, cos_nu)

    # Argument of latitude
    phi_k = nu_k + omega

    # Second-harmonic perturbation corrections
    sin2phi = np.sin(2.0 * phi_k)
    cos2phi = np.cos(2.0 * phi_k)
    delta_u = Cus * sin2phi + Cuc * cos2phi
    delta_r = Crs * sin2phi + Crc * cos2phi
    delta_i = Cis * sin2phi + Cic * cos2phi

    # Corrected values
    u_k = phi_k + delta_u
    r_k = a * (1.0 - e * np.cos(Ek)) + delta_r
    i_k = i0 + delta_i + idot * tk

    # Position in orbital plane
    x_prime = r_k * np.cos(u_k)
    y_prime = r_k * np.sin(u_k)

    # Corrected longitude of ascending node
    OMEGA_k = (OMEGA0
               + (OMEGA_dot - OMEGA_E) * tk
               - OMEGA_E * toe)

    cos_OMEGA = np.cos(OMEGA_k)
    sin_OMEGA = np.sin(OMEGA_k)
    cos_i     = np.cos(i_k)
    sin_i     = np.sin(i_k)

    # ECEF coordinates (metres)
    x_m = x_prime * cos_OMEGA - y_prime * cos_i * sin_OMEGA
    y_m = x_prime * sin_OMEGA + y_prime * cos_i * cos_OMEGA
    z_m = y_prime * sin_i

    return np.array([x_m, y_m, z_m])


def _epoch_to_gps_sow(epoch: pd.Timestamp) -> float:
    """Convert a UTC epoch timestamp to GPS seconds-of-week."""
    if epoch.tzinfo is None:
        epoch = epoch.tz_localize("UTC")
    total_s = (epoch - _GPS_EPOCH).total_seconds()
    return total_s % GPS_SEC_PER_WEEK


# ── 15-min resampling (zero-order hold) ──────────────────────────────────────

def _resample_to_15min(
    sat_id: str,
    sise_series: pd.Series,
    full_grid: pd.DatetimeIndex,
    max_fill_steps: int,
) -> pd.DataFrame:
    """
    Resample an irregular broadcast-epoch Series to the 15-min grid.

    Strategy:
    1. Drop duplicate epochs (keep last — most recent upload wins)
    2. Reindex to the 15-min grid (introduces NaN at non-broadcast times)
    3. Forward-fill (zero-order hold) — use the most recently uploaded ephemeris
    4. Cap fill at max_fill_steps consecutive steps (leave beyond that as NaN)

    Parameters
    ----------
    sat_id : str
    sise_series : pd.Series — irregular timestamps, sise_ns values
    full_grid : pd.DatetimeIndex — the target 15-min grid
    max_fill_steps : int — max consecutive ffill steps

    Returns
    -------
    pd.DataFrame  with columns: satellite_id | clock_error_ns | eph_error_m | sise_ns
    """
    # Convert index to proper DatetimeIndex
    idx = pd.DatetimeIndex(sise_series.index).tz_localize("UTC") \
          if sise_series.index.tz is None else pd.DatetimeIndex(sise_series.index)

    s = pd.Series(sise_series.values, index=idx, name="sise_ns")

    # Drop duplicates at same epoch (keep last — newest upload)
    s = s[~s.index.duplicated(keep="last")].sort_index()

    # Reindex to full 15-min grid
    s = s.reindex(full_grid)

    # Zero-order hold: forward fill, capped at max_fill_steps
    s = s.ffill(limit=max_fill_steps)

    n_valid = s.notna().sum()
    n_total = len(s)
    logger.debug(
        f"[{sat_id}] Resampled to 15-min: {n_valid}/{n_total} valid "
        f"({n_valid/n_total:.1%})"
    )

    # Build output DataFrame matching data_loader.py format
    out = pd.DataFrame({
        "satellite_id":   sat_id,
        "clock_error_ns": s.values,   # proxy or IGS-based clock error
        "eph_error_m":    np.nan,      # only populated for GPS Tier A below
        "sise_ns":        s.values,
    }, index=full_grid)
    out.index.name = "timestamp"
    return out


def _save_intermediate_csv(
    sat_id: str,
    df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Save per-satellite intermediate CSV for data_loader.py to consume."""
    df_out = df.reset_index()
    df_out.to_csv(out_path, index=False)
    logger.debug(f"[{sat_id}] Saved {len(df_out)} rows → {out_path.name}")


# ── Raw CSV loading ───────────────────────────────────────────────────────────

def _load_competition_csvs(
    data_dir: Path,
    pattern: str,
    n_files: int,
) -> pd.DataFrame:
    """
    Load the first n_files CSVs matching pattern from data_dir.
    Files are sorted lexicographically (2026_001 < 2026_002 < …).

    If pattern is an exact filename (no glob wildcards), all matching files
    are loaded regardless of n_files — useful for processing a single day.
    """
    is_exact = "*" not in pattern and "?" not in pattern and "[" not in pattern

    all_files = sorted(data_dir.glob(pattern))
    if not all_files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in {data_dir}.\n"
            f"Expected files like 2026_001.csv … 2026_007.csv."
        )

    train_files = all_files if is_exact else all_files[:n_files]
    logger.info(f"[Module 0] Using {len(train_files)} file(s): "
                f"{[f.name for f in train_files]}")

    frames: List[pd.DataFrame] = []
    for f in train_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            frames.append(df)
            logger.debug(f"  Loaded {f.name}: {len(df):,} rows")
        except Exception as exc:
            logger.error(f"  Failed to load {f.name}: {exc}")

    if not frames:
        raise ValueError("Could not load any competition CSV files.")

    combined = pd.concat(frames, ignore_index=True)
    logger.debug(f"  Combined: {len(combined):,} rows, {len(combined.columns)} columns")
    return combined


# ── CLI entry point (standalone use) ─────────────────────────────────────────

def _parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="Module 0: Preprocess competition CSVs → per-satellite SISE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert 7 training days, write to data/processed/sise_intermediate/
  python src/gnss_preprocess.py --data data/raw/train/ --output data/processed/sise_intermediate/

  # Force recompute (ignore cached output CSVs)
  python src/gnss_preprocess.py --data data/raw/train/ --output data/processed/sise_intermediate/ --force
        """,
    )
    p.add_argument("--data",   required=True,     help="Dir containing 2026_NNN.csv files")
    p.add_argument("--output", required=True,     help="Dir for per-satellite _sise.csv files")
    p.add_argument("--days",   type=int, default=7, help="Number of training day files to use")
    p.add_argument("--force",  action="store_true", help="Recompute even if output exists")
    p.add_argument("--verbose", action="store_true", help="Debug logging")
    return p.parse_args()


if __name__ == "__main__":
    import logging as _logging
    args = _parse_args()
    _logging.basicConfig(
        level=_logging.DEBUG if args.verbose else _logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    prepare_pipeline_input(
        data_dir=args.data,
        output_dir=args.output,
        n_train_days=args.days,
        force_recompute=args.force,
    )
