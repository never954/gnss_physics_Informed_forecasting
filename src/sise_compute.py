"""
sise_compute.py — Module 3: Compute SISE from raw RINEX + IGS SP3 files.

This module is OPTIONAL — only needed if the competition provides raw
navigation RINEX files and IGS precise clock/orbit files instead of
pre-computed errors.

SISE (Signal-in-Space Error) = broadcast_state − igs_truth

Clock component:
    SISE_clk = (broadcast_clock_correction − igs_clock) × c   [in ns]

Ephemeris (radial) component:
    SISE_eph = radial_component(broadcast_pos − igs_pos)      [in m]

Combined (simplified):
    SISE = sqrt(SISE_clk² + (SISE_eph/c)²)                   [in ns]

Dependencies: georinex, astropy (see requirements.txt)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Speed of light
C_M_PER_S: float = 299_792_458.0   # m/s
C_M_PER_NS: float = C_M_PER_S / 1e9  # 0.2998 m/ns


def compute_sise(
    broadcast_rinex_dir: str | Path,
    igs_clock_dir: str | Path,
    igs_sp3_dir: str | Path,
    output_csv: str | Path,
    interval_min: int = 15,
) -> pd.DataFrame:
    """
    Compute SISE from broadcast RINEX and IGS precise products.

    Parameters
    ----------
    broadcast_rinex_dir : str | Path
        Directory containing navigation RINEX files (*.rnx, *.n, *brdc*.*)
    igs_clock_dir : str | Path
        Directory containing IGS clock RINEX files (*.clk)
    igs_sp3_dir : str | Path
        Directory containing IGS SP3 precise orbit files (*.sp3)
    output_csv : str | Path
        Path where the computed error CSV will be written.
    interval_min : int
        Sampling interval in minutes (default 15).

    Returns
    -------
    pd.DataFrame
        With columns: timestamp, sat_id, clock_error_ns, eph_error_m, sise_ns
    """
    try:
        import georinex as gr
    except ImportError:
        raise ImportError(
            "georinex is required for SISE computation. "
            "Install with: pip install georinex"
        )

    broadcast_rinex_dir = Path(broadcast_rinex_dir)
    igs_clock_dir = Path(igs_clock_dir)
    igs_sp3_dir = Path(igs_sp3_dir)
    output_csv = Path(output_csv)

    logger.info("Starting SISE computation from RINEX files...")

    # --- Load broadcast navigation data ---
    nav_files = list(broadcast_rinex_dir.glob("*.rnx")) + \
                list(broadcast_rinex_dir.glob("*.n")) + \
                list(broadcast_rinex_dir.glob("*brdc*"))
    if not nav_files:
        raise FileNotFoundError(f"No navigation RINEX files found in {broadcast_rinex_dir}")

    logger.info(f"Loading {len(nav_files)} navigation RINEX file(s)...")
    nav_data = []
    for f in nav_files:
        try:
            nav = gr.load(f)
            nav_data.append(nav)
            logger.debug(f"  Loaded {f.name}")
        except Exception as e:
            logger.warning(f"  Failed to load {f.name}: {e}")

    if not nav_data:
        raise ValueError("Could not load any navigation RINEX files.")

    # --- Load IGS SP3 precise orbits ---
    sp3_files = sorted(igs_sp3_dir.glob("*.sp3"))
    if not sp3_files:
        raise FileNotFoundError(f"No SP3 files found in {igs_sp3_dir}")

    logger.info(f"Loading {len(sp3_files)} SP3 file(s)...")
    sp3_data = []
    for f in sp3_files:
        try:
            sp3 = gr.load(f)
            sp3_data.append(sp3)
            logger.debug(f"  Loaded {f.name}")
        except Exception as e:
            logger.warning(f"  Failed to load {f.name}: {e}")

    # --- Load IGS clock files ---
    clk_files = sorted(igs_clock_dir.glob("*.clk"))
    if not clk_files:
        raise FileNotFoundError(f"No IGS clock files found in {igs_clock_dir}")

    logger.info(f"Loading {len(clk_files)} clock file(s)...")
    clk_data = []
    for f in clk_files:
        try:
            clk = gr.load(f)
            clk_data.append(clk)
            logger.debug(f"  Loaded {f.name}")
        except Exception as e:
            logger.warning(f"  Failed to load {f.name}: {e}")

    # --- Compute differences ---
    rows = []
    logger.info("Computing SISE differences (broadcast − IGS truth)...")
    # NOTE: Actual implementation depends on georinex Dataset structure.
    # The logic below is a blueprint — adapt column names to your RINEX version.

    logger.warning(
        "SISE computation from RINEX is dataset-specific. "
        "This function provides the structural framework. "
        "You may need to adapt the column extraction to your specific RINEX format. "
        "If competition provides pre-computed errors, use data_loader.py directly."
    )

    # Placeholder result — replace with actual computation
    result = pd.DataFrame(rows, columns=["timestamp", "sat_id", "clock_error_ns", "eph_error_m", "sise_ns"])
    result.to_csv(output_csv, index=False)
    logger.info(f"SISE written to {output_csv}")
    return result


def compute_clock_sise(
    broadcast_clock_s: float,
    igs_clock_s: float,
) -> float:
    """
    Compute clock component of SISE in nanoseconds.

    Parameters
    ----------
    broadcast_clock_s : float
        Broadcast clock correction in seconds.
    igs_clock_s : float
        IGS precise clock correction in seconds.

    Returns
    -------
    float : clock SISE in nanoseconds
    """
    delta_s = broadcast_clock_s - igs_clock_s
    return delta_s * 1e9  # Convert seconds → nanoseconds


def compute_radial_sise(
    broadcast_pos_ecef: np.ndarray,
    igs_pos_ecef: np.ndarray,
    sat_pos_ecef: np.ndarray,
) -> float:
    """
    Compute the radial (line-of-sight) component of ephemeris SISE.

    The radial direction is the unit vector from Earth centre to satellite.
    Only the radial component matters for ranging error (cross-track and
    along-track are partially cancelled by geometry).

    Parameters
    ----------
    broadcast_pos_ecef : np.ndarray shape (3,)
        Satellite position from broadcast ephemeris [m, ECEF].
    igs_pos_ecef : np.ndarray shape (3,)
        Satellite position from IGS SP3 [m, ECEF].
    sat_pos_ecef : np.ndarray shape (3,)
        Reference satellite position (use IGS) for radial unit vector.

    Returns
    -------
    float : radial ephemeris error in metres
    """
    # Radial unit vector (Earth centre → satellite)
    r_hat = sat_pos_ecef / np.linalg.norm(sat_pos_ecef)
    pos_diff = broadcast_pos_ecef - igs_pos_ecef
    return float(np.dot(pos_diff, r_hat))


def broadcast_clock_at_time(
    t: float,
    t_oc: float,
    a0: float,
    a1: float,
    a2: float,
    relativistic_correction: float = 0.0,
) -> float:
    """
    Evaluate the broadcast clock polynomial.

    Δt_sv = a0 + a1·(t - t_oc) + a2·(t - t_oc)²

    Relativistic correction (already in broadcast for GPS):
    F · e · sqrt(A) · sin(E_k)
    where F = -4.442807633e-10 s/√m

    Parameters
    ----------
    t : float
        Current GPS time (seconds).
    t_oc : float
        Clock reference time from navigation message (seconds).
    a0, a1, a2 : float
        Clock coefficients from navigation message.
    relativistic_correction : float
        Pre-computed relativistic term (seconds). Already in broadcast for GPS.

    Returns
    -------
    float : satellite clock correction in seconds
    """
    dt = t - t_oc
    # Account for week rollover
    if dt > 302400:
        dt -= 604800
    elif dt < -302400:
        dt += 604800

    return a0 + a1 * dt + a2 * dt**2 + relativistic_correction
