"""
classifier.py — Module 4: Classify each satellite by orbit type and reset pattern.

Level 1 — Orbital classification:
    GEO/GSO : stationary relative to Earth, dominant 24hr error cycle
    MEO     : orbits every ~12–14hr, dominant orbital-period error cycle

Level 2 — Reset pattern classification (on detrended residual):
    clean    : < 2 detected resets
    regular  : ≥2 resets with consistent timing (predictable uploads)
    irregular: ≥2 resets with erratic timing (needs robust model)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    SAT_PREFIX_MAP,
    GEO_SAT_IDS,
    BEIDOU_GEO_PRNS,
    BEIDOU_IGSO_PRNS,
    ORBITAL_PERIODS_HR,
    MIN_RESETS_FOR_PATTERN,
    SAWTOOTH_INTERVAL_MAX_HR,
    PREDICTION_INTERVAL_MIN,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OrbitType(str, Enum):
    GEO = "GEO"   # Includes IGSO — treated same way (dominant 24hr cycle)
    MEO = "MEO"


class ResetPattern(str, Enum):
    CLEAN     = "clean"
    REGULAR   = "regular"
    IRREGULAR = "irregular"


class ModelType(str, Enum):
    GP          = "GP"           # 5-kernel GP (clean)
    BOOTSTRAP   = "BootstrapMC"  # Bootstrap Monte Carlo (regular sawtooth)
    STUDENT_T   = "StudentT"     # Student-t Process (irregular sawtooth)
    MATERN      = "Matern"       # Fallback


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SatelliteClassification:
    sat_id: str
    orbit_type: OrbitType
    constellation: str
    orbital_period_hr: float        # For kernel initialization
    reset_pattern: ResetPattern
    n_resets: int
    mean_reset_interval_hr: float
    std_reset_interval_hr: float
    model_type: ModelType
    fft_dominant_period_hr: Optional[float] = None  # Set if FFT was used
    notes: str = ""

    def to_dict(self) -> Dict:
        return {
            "sat_id": self.sat_id,
            "orbit_type": self.orbit_type.value,
            "constellation": self.constellation,
            "orbital_period_hr": self.orbital_period_hr,
            "reset_pattern": self.reset_pattern.value,
            "n_resets": self.n_resets,
            "mean_reset_interval_hr": self.mean_reset_interval_hr,
            "std_reset_interval_hr": self.std_reset_interval_hr,
            "model_type": self.model_type.value,
            "fft_dominant_period_hr": self.fft_dominant_period_hr,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_satellite(
    sat_id: str,
    residual: pd.Series,
    resets: List,  # List[ResetEvent] from reset_detector.py
    use_fft_fallback: bool = True,
) -> SatelliteClassification:
    """
    Classify one satellite and determine its model routing.

    Parameters
    ----------
    sat_id : str
        Satellite PRN (e.g., "G01", "C03", "E05").
    residual : pd.Series
        Detrended residual time series (from detrend.py).
    resets : list of ResetEvent
        Detected reset events (from reset_detector.py).
    use_fft_fallback : bool
        If True, use FFT to estimate orbital period when constellation unknown.

    Returns
    -------
    SatelliteClassification
    """
    # --- Level 1: Orbit type ---
    orbit_type, constellation, orbital_period_hr = _classify_orbit(
        sat_id, residual, use_fft_fallback
    )

    # --- Level 2: Reset pattern ---
    reset_pattern, n_resets, mean_interval_hr, std_interval_hr = _classify_resets(resets)

    # --- Model routing ---
    model_type = _route_model(reset_pattern)

    classification = SatelliteClassification(
        sat_id=sat_id,
        orbit_type=orbit_type,
        constellation=constellation,
        orbital_period_hr=orbital_period_hr,
        reset_pattern=reset_pattern,
        n_resets=n_resets,
        mean_reset_interval_hr=mean_interval_hr,
        std_reset_interval_hr=std_interval_hr,
        model_type=model_type,
    )

    logger.info(
        f"[{sat_id}] Orbit={orbit_type.value} | Constellation={constellation} | "
        f"Period={orbital_period_hr:.2f}hr | Resets={n_resets} ({reset_pattern.value}) | "
        f"Model={model_type.value}"
    )
    return classification


def classify_all(
    sat_data: Dict[str, pd.DataFrame],
    residuals: Dict[str, pd.Series],
    resets_map: Dict[str, List],
) -> Dict[str, SatelliteClassification]:
    """
    Classify all satellites in the dataset.

    Parameters
    ----------
    sat_data : Dict[sat_id, DataFrame] — raw data (for coverage checks)
    residuals : Dict[sat_id, Series] — detrended residuals
    resets_map : Dict[sat_id, List[ResetEvent]] — detected resets per satellite

    Returns
    -------
    Dict[sat_id, SatelliteClassification]
    """
    results = {}
    for sat_id in sat_data:
        residual = residuals.get(sat_id, pd.Series(dtype=float))
        resets = resets_map.get(sat_id, [])
        results[sat_id] = classify_satellite(sat_id, residual, resets)

    # Summary log
    model_counts = {}
    for c in results.values():
        model_counts[c.model_type.value] = model_counts.get(c.model_type.value, 0) + 1
    logger.info(f"Classification summary: {model_counts}")
    return results


def classification_report(
    classifications: Dict[str, SatelliteClassification],
) -> pd.DataFrame:
    """Return a DataFrame summary of all classifications (for logging / output)."""
    rows = [c.to_dict() for c in classifications.values()]
    return pd.DataFrame(rows).set_index("sat_id")


# ---------------------------------------------------------------------------
# Level 1: Orbit classification
# ---------------------------------------------------------------------------

def _classify_orbit(
    sat_id: str,
    residual: pd.Series,
    use_fft_fallback: bool,
) -> Tuple[OrbitType, str, float]:
    """
    Returns (orbit_type, constellation_name, orbital_period_hr).
    """
    # --- Step 1: ID-based lookup (fast, deterministic) ---
    if sat_id in GEO_SAT_IDS:
        return OrbitType.GEO, _get_constellation(sat_id), ORBITAL_PERIODS_HR["BEIDOU_GEO"]

    # Known SBAS / QZSS (always GEO)
    if sat_id.startswith(("S", "J")):
        return OrbitType.GEO, "SBAS", ORBITAL_PERIODS_HR["SBAS"]

    # All others: look up constellation, assume MEO
    constellation = _get_constellation(sat_id)
    orbital_period_hr = ORBITAL_PERIODS_HR.get(constellation, ORBITAL_PERIODS_HR["UNKNOWN"])

    if constellation != "UNKNOWN":
        return OrbitType.MEO, constellation, orbital_period_hr

    # --- Step 2: FFT fallback for unknown constellations ---
    if use_fft_fallback and len(residual) >= 96:
        fft_period = _estimate_dominant_period_fft(residual)
        if fft_period is not None:
            logger.info(f"[{sat_id}] FFT estimated dominant period: {fft_period:.2f} hr")
            if fft_period > 20.0:
                return OrbitType.GEO, "UNKNOWN_GEO", fft_period
            else:
                return OrbitType.MEO, "UNKNOWN_MEO", fft_period

    logger.warning(f"[{sat_id}] Could not determine orbit type. Defaulting to MEO.")
    return OrbitType.MEO, "UNKNOWN", ORBITAL_PERIODS_HR["UNKNOWN"]


def _get_constellation(sat_id: str) -> str:
    """Map satellite ID prefix to constellation name."""
    if not sat_id:
        return "UNKNOWN"
    prefix = sat_id[0].upper()

    # BeiDou: distinguish GEO/IGSO vs MEO by number
    if prefix == "C":
        try:
            prn_num = int(sat_id[1:])
            if prn_num <= 5:
                return "BEIDOU_GEO"
            elif prn_num <= 16:
                return "BEIDOU_GEO"   # IGSO treated same as GEO
            else:
                return "BEIDOU_MEO"
        except ValueError:
            return "BEIDOU_MEO"

    return SAT_PREFIX_MAP.get(prefix, "UNKNOWN")


def _estimate_dominant_period_fft(residual: pd.Series) -> Optional[float]:
    """
    Use FFT to estimate the dominant periodic component in the residual.
    Returns the period in hours, or None if no clear peak found.
    """
    values = residual.dropna().values
    if len(values) < 48:
        return None

    # Sampling rate: one sample per 15 min = 4 per hour
    samples_per_hour = 60 / PREDICTION_INTERVAL_MIN

    fft_vals = np.abs(np.fft.rfft(values - values.mean()))
    freqs = np.fft.rfftfreq(len(values), d=1.0 / samples_per_hour)  # in cycles/hour

    # Exclude DC (index 0) and very high frequencies
    valid = (freqs > 0.02) & (freqs < 0.5)   # 2hr < period < 50hr
    if not valid.any():
        return None

    peak_idx = np.argmax(fft_vals[valid])
    dominant_freq = freqs[valid][peak_idx]

    if dominant_freq < 1e-6:
        return None
    return 1.0 / dominant_freq  # Convert frequency → period in hours


# ---------------------------------------------------------------------------
# Level 2: Reset pattern classification
# ---------------------------------------------------------------------------

def _classify_resets(
    resets: List,  # List[ResetEvent]
) -> Tuple[ResetPattern, int, float, float]:
    """
    Returns (pattern, n_resets, mean_interval_hr, std_interval_hr).
    """
    # Filter to real resets only (not eclipses)
    real_resets = [
        r for r in resets
        if getattr(getattr(r, "event_type", "reset"), "value", getattr(r, "event_type", "reset")) == "reset"
    ]
    n = len(real_resets)

    if n < MIN_RESETS_FOR_PATTERN:
        return ResetPattern.CLEAN, n, 0.0, 0.0

    # Compute inter-reset intervals
    if n == 1:
        return ResetPattern.CLEAN, n, 0.0, 0.0

    timestamps = sorted([getattr(r, "timestamp", 0) for r in real_resets])
    intervals_hr = [
        (timestamps[i+1] - timestamps[i]).total_seconds() / 3600
        if hasattr(timestamps[i+1] - timestamps[i], "total_seconds")
        else (timestamps[i+1] - timestamps[i]) / 3600
        for i in range(len(timestamps) - 1)
    ]

    mean_hr = float(np.mean(intervals_hr))
    std_hr = float(np.std(intervals_hr))

    # Coefficient of variation: irregular if timing variance > 40% of mean
    cv = std_hr / max(mean_hr, 1e-6)
    if mean_hr <= SAWTOOTH_INTERVAL_MAX_HR and cv < 0.4:
        return ResetPattern.REGULAR, n, mean_hr, std_hr
    else:
        return ResetPattern.IRREGULAR, n, mean_hr, std_hr


# ---------------------------------------------------------------------------
# Model routing (purely from reset pattern — orbit type handled inside models)
# ---------------------------------------------------------------------------

def _route_model(reset_pattern: ResetPattern) -> ModelType:
    """Map reset pattern to model architecture."""
    return {
        ResetPattern.CLEAN:     ModelType.GP,
        ResetPattern.REGULAR:   ModelType.BOOTSTRAP,
        ResetPattern.IRREGULAR: ModelType.STUDENT_T,
    }[reset_pattern]
