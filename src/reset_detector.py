"""
reset_detector.py — Module 6: Detect sawtooth resets and eclipse spikes.

Algorithm:
1. Compute the detrended residual's MAD-based outlier score
2. Flag any step where |residual - median| > MAD_THRESHOLD × MAD as a spike
3. Eclipse filter: if the spike recovers within ECLIPSE_RECOVERY_STEPS steps
   back towards the pre-spike level → mark as ECLIPSE and skip
4. Otherwise: sustained shift → RESET, record timestamp + magnitude

Why MAD instead of z-score?
   MAD is robust to outliers themselves. A z-score can be pulled by the
   very jumps we're trying to detect. MAD stays stable.

Why 3-step recovery window for eclipses?
   Eclipse thermal transients for GPS/Galileo typically decay in 30–45 min
   (2–3 samples at 15-min cadence) back towards the pre-eclipse level.
   Resets from ground uploads cause a permanent level shift.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    MAD_THRESHOLD_SIGMA,
    ECLIPSE_RECOVERY_STEPS,
    ECLIPSE_RECOVERY_FRACTION,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    RESET   = "reset"    # Ground upload correction (sustained level shift)
    ECLIPSE = "eclipse"  # Thermal spike from entering/leaving shadow (transient)


@dataclass
class ResetEvent:
    timestamp: pd.Timestamp
    index: int                    # Position in the residual series
    magnitude_ns: float           # Size of jump in nanoseconds (signed)
    event_type: EventType
    recovery_steps: Optional[int] = None  # For eclipses: steps to recover

    def __repr__(self) -> str:
        return (
            f"ResetEvent({self.event_type.value}, "
            f"t={self.timestamp}, "
            f"mag={self.magnitude_ns:.2f}ns)"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_resets(
    residual: pd.Series,
    sat_id: str = "UNKNOWN",
    threshold_sigma: float = MAD_THRESHOLD_SIGMA,
    eclipse_recovery_steps: int = ECLIPSE_RECOVERY_STEPS,
    eclipse_recovery_fraction: float = ECLIPSE_RECOVERY_FRACTION,
    min_gap_between_events: int = 2,
) -> List[ResetEvent]:
    """
    Detect sawtooth resets and eclipse spikes in a detrended residual.

    Parameters
    ----------
    residual : pd.Series
        Detrended residual (ns), indexed by DatetimeIndex.
    sat_id : str
        Satellite ID for logging.
    threshold_sigma : float
        Number of MAD units to flag as a jump (default 3.0).
    eclipse_recovery_steps : int
        Max steps within which a spike must recover to be classified as eclipse.
    eclipse_recovery_fraction : float
        Spike must return to within this fraction of pre-spike amplitude to qualify.
    min_gap_between_events : int
        Minimum number of steps between consecutive events (prevents double-counting).

    Returns
    -------
    List[ResetEvent]
        Sorted by timestamp, with event_type = RESET or ECLIPSE.
    """
    if residual.isna().all():
        logger.warning(f"[{sat_id}] All residual values are NaN. No resets detected.")
        return []

    values = residual.ffill().bfill().values
    timestamps = residual.index
    n = len(values)

    if n < 10:
        logger.warning(f"[{sat_id}] Too few points ({n}) for reset detection.")
        return []

    # --- Step 1: Compute MAD-based outlier score on first-differences ---
    # First-differences highlight sudden jumps more clearly than raw values
    diffs = np.diff(values, prepend=values[0])
    median_diff = np.median(diffs)
    mad = _compute_mad(diffs)

    if mad < 1e-10:
        logger.debug(f"[{sat_id}] MAD ≈ 0 — no significant variability in residual diffs.")
        return []

    # Z-score in MAD units
    mad_z = np.abs((diffs - median_diff) / (1.4826 * mad))  # 1.4826 for normal consistency

    # --- Step 2: Find candidate spike positions ---
    spike_mask = mad_z > threshold_sigma
    spike_indices = np.where(spike_mask)[0]

    if len(spike_indices) == 0:
        logger.debug(f"[{sat_id}] No jumps detected above {threshold_sigma}σ threshold.")
        return []

    logger.debug(f"[{sat_id}] Found {len(spike_indices)} candidate spike(s) before eclipse filter.")

    # --- Step 3: Classify each spike as RESET or ECLIPSE ---
    events: List[ResetEvent] = []
    last_event_idx = -min_gap_between_events - 1

    for idx in spike_indices:
        # Enforce minimum gap between events
        if idx - last_event_idx < min_gap_between_events:
            continue

        magnitude = diffs[idx]
        ts = timestamps[idx]

        # Eclipse check: does the signal recover within `eclipse_recovery_steps` steps?
        event_type = _classify_spike(
            values=values,
            spike_idx=idx,
            magnitude=magnitude,
            recovery_steps=eclipse_recovery_steps,
            recovery_fraction=eclipse_recovery_fraction,
        )

        recovery_steps_count = None
        if event_type == EventType.ECLIPSE:
            # Count actual recovery steps
            recovery_steps_count = _count_recovery_steps(
                values, idx, magnitude, eclipse_recovery_steps, eclipse_recovery_fraction
            )

        events.append(ResetEvent(
            timestamp=ts,
            index=idx,
            magnitude_ns=float(magnitude),
            event_type=event_type,
            recovery_steps=recovery_steps_count,
        ))
        last_event_idx = idx

    n_resets   = sum(1 for e in events if e.event_type == EventType.RESET)
    n_eclipses = sum(1 for e in events if e.event_type == EventType.ECLIPSE)

    logger.info(
        f"[{sat_id}] Reset detection: {n_resets} reset(s), {n_eclipses} eclipse(s) | "
        f"MAD={1.4826*mad:.3f} ns | threshold={threshold_sigma}σ"
    )

    return sorted(events, key=lambda e: e.timestamp)


def reset_statistics(events: List[ResetEvent]) -> dict:
    """
    Compute statistical summary of detected resets.
    Used by Bootstrap MC model for reset resampling.

    Returns
    -------
    dict with keys:
        n_resets, mean_magnitude_ns, std_magnitude_ns,
        mean_interval_hr, std_interval_hr, magnitudes, intervals_hr
    """
    resets = [e for e in events if e.event_type == EventType.RESET]
    n = len(resets)

    if n == 0:
        return {
            "n_resets": 0,
            "mean_magnitude_ns": 0.0,
            "std_magnitude_ns": 0.0,
            "mean_interval_hr": float("inf"),
            "std_interval_hr": 0.0,
            "magnitudes": [],
            "intervals_hr": [],
        }

    magnitudes = [r.magnitude_ns for r in resets]
    intervals_hr = []
    if n > 1:
        for i in range(n - 1):
            dt = (resets[i+1].timestamp - resets[i].timestamp).total_seconds() / 3600
            intervals_hr.append(dt)

    return {
        "n_resets": n,
        "mean_magnitude_ns": float(np.mean(magnitudes)),
        "std_magnitude_ns": float(np.std(magnitudes)),
        "mean_interval_hr": float(np.mean(intervals_hr)) if intervals_hr else float("inf"),
        "std_interval_hr": float(np.std(intervals_hr)) if intervals_hr else 0.0,
        "magnitudes": magnitudes,
        "intervals_hr": intervals_hr,
    }


def mask_eclipses(residual: pd.Series, events: List[ResetEvent]) -> pd.Series:
    """
    Set eclipse-contaminated timestamps to NaN in the residual.
    Eclipse-affected points confuse GP fitting — better to interpolate over them.
    """
    residual = residual.copy()
    eclipse_events = [e for e in events if e.event_type == EventType.ECLIPSE]
    for eclipse in eclipse_events:
        # Mask the spike and recovery window
        n_steps = eclipse.recovery_steps or ECLIPSE_RECOVERY_STEPS
        start_idx = eclipse.index
        end_idx   = min(eclipse.index + n_steps + 1, len(residual))
        residual.iloc[start_idx:end_idx] = np.nan
    return residual


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_mad(values: np.ndarray) -> float:
    """Median Absolute Deviation — robust measure of scale."""
    return float(np.median(np.abs(values - np.median(values))))


def _classify_spike(
    values: np.ndarray,
    spike_idx: int,
    magnitude: float,
    recovery_steps: int,
    recovery_fraction: float,
) -> EventType:
    """
    Determine if a spike is a transient eclipse or a sustained reset.

    Logic:
    - Pre-spike baseline: median of 5 points before the spike
    - Look at the next `recovery_steps` points
    - If the signal returns within `recovery_fraction` of the spike amplitude
      back towards the baseline → ECLIPSE (transient)
    - Otherwise → RESET (sustained shift)
    """
    n = len(values)
    baseline_start = max(0, spike_idx - 5)
    baseline_end   = spike_idx
    if baseline_end <= baseline_start:
        return EventType.RESET

    pre_spike_level = np.median(values[baseline_start:baseline_end])
    spike_level     = values[spike_idx] if spike_idx < n else pre_spike_level

    # Amplitude of deviation from baseline
    amplitude = abs(spike_level - pre_spike_level)
    if amplitude < 1e-10:
        return EventType.RESET

    # Check recovery in subsequent steps
    for step in range(1, recovery_steps + 1):
        check_idx = spike_idx + step
        if check_idx >= n:
            break
        current_deviation = abs(values[check_idx] - pre_spike_level)
        if current_deviation < recovery_fraction * amplitude:
            return EventType.ECLIPSE

    return EventType.RESET


def _count_recovery_steps(
    values: np.ndarray,
    spike_idx: int,
    magnitude: float,
    max_recovery_steps: int,
    recovery_fraction: float,
) -> int:
    """Count how many steps it takes for the spike to recover."""
    n = len(values)
    baseline_start = max(0, spike_idx - 5)
    pre_spike_level = np.median(values[baseline_start:spike_idx])
    spike_level = values[spike_idx] if spike_idx < n else pre_spike_level
    amplitude = abs(spike_level - pre_spike_level)

    if amplitude < 1e-10:
        return 0

    for step in range(1, max_recovery_steps + 1):
        check_idx = spike_idx + step
        if check_idx >= n:
            return step
        if abs(values[check_idx] - pre_spike_level) < recovery_fraction * amplitude:
            return step

    return max_recovery_steps
