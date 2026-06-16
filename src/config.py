"""
config.py — Central configuration for the GNSS SISE Prediction Pipeline.

All physical constants, model hyperparameters, and data-format aliases
live here. Edit this file on competition day if column names differ.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# 1. CONSTELLATION ORBITAL PERIODS
# ---------------------------------------------------------------------------
# True (non-rounded) orbital repeat periods in HOURS.
# Used as the centre of the learnable periodic kernel.
# Source: IGS satellite metadata + ICD documents.

ORBITAL_PERIODS_HR: Dict[str, float] = {
    "GPS":     11.9667,   # 11h 58m  (GPS Block II/IIF/III)
    "GALILEO": 14.0833,   # 14h 05m
    "BEIDOU_MEO": 12.8833, # 12h 53m  (BeiDou MEO C19+)
    "BEIDOU_GEO": 24.0,   # Geostationary — treat as 24h
    "GLONASS": 11.2667,   # 11h 16m
    "SBAS":    24.0,      # Geostationary SBAS (WAAS, EGNOS, GAGAN, MSAS)
    "UNKNOWN": 12.0,      # Fallback
}

# Allowed search range: ±30% of centre value (enforced as GP kernel constraint)
ORBITAL_PERIOD_TOLERANCE: float = 0.30  # 30%

# Solar / diurnal period
SOLAR_PERIOD_HR: float = 24.0
SOLAR_PERIOD_MIN_HR: float = 18.0
SOLAR_PERIOD_MAX_HR: float = 30.0

# ---------------------------------------------------------------------------
# 2. SATELLITE ID → CONSTELLATION MAPPING
# ---------------------------------------------------------------------------
# Prefix rules for PRN/SV IDs as typically seen in RINEX / IGS files.
# Key: prefix string  Value: constellation name (must match ORBITAL_PERIODS_HR)

SAT_PREFIX_MAP: Dict[str, str] = {
    "G": "GPS",
    "E": "GALILEO",
    "C": "BEIDOU_MEO",   # refined to GEO below
    "R": "GLONASS",
    "S": "SBAS",
    "J": "SBAS",         # QZSS — treat as GEO for now
}

# BeiDou GEO satellites (C01–C05 are GEO/IGSO; C06–C16 IGSO; C19+ MEO)
BEIDOU_GEO_PRNS: List[str] = [f"C{i:02d}" for i in range(1, 6)]   # C01–C05
BEIDOU_IGSO_PRNS: List[str] = [f"C{i:02d}" for i in range(6, 17)]  # C06–C16 (treat as GEO)

# GEO satellite IDs (any constellation prefix that maps to GEO orbit)
GEO_SAT_IDS: List[str] = BEIDOU_GEO_PRNS + BEIDOU_IGSO_PRNS + ["S20", "S25", "S48"]

# ---------------------------------------------------------------------------
# 3. CLOCK TYPE → KERNEL RANGE MAPPING
# ---------------------------------------------------------------------------
# Hydrogen Maser: very stable → long-range kernel
# Rubidium:       moderate   → medium-range kernel
# Cesium:         noisy      → short-range kernel
#
# The RBF (slow-trend) kernel length-scale is initialized from these values.

@dataclass
class ClockKernelConfig:
    rbf_range_hr: float        # RBF length-scale initial value in hours
    rbf_range_bounds: Tuple[float, float]  # (min, max) in hours

CLOCK_KERNEL_CONFIGS: Dict[str, ClockKernelConfig] = {
    "H-MASER": ClockKernelConfig(rbf_range_hr=80.0,  rbf_range_bounds=(40.0, 200.0)),
    "RUBIDIUM": ClockKernelConfig(rbf_range_hr=50.0, rbf_range_bounds=(20.0, 120.0)),
    "CESIUM":   ClockKernelConfig(rbf_range_hr=30.0, rbf_range_bounds=(10.0,  80.0)),
    "UNKNOWN":  ClockKernelConfig(rbf_range_hr=50.0, rbf_range_bounds=(15.0, 120.0)),
}

# Satellite ID → clock type (add entries as catalog becomes known)
SAT_CLOCK_MAP: Dict[str, str] = {
    # GPS Block III (H-Maser)
    "G01": "RUBIDIUM", "G02": "RUBIDIUM", "G03": "RUBIDIUM",
    "G11": "RUBIDIUM", "G14": "RUBIDIUM", "G18": "RUBIDIUM",
    # Galileo (H-Maser)
    "E01": "H-MASER", "E02": "H-MASER", "E03": "H-MASER",
    "E04": "H-MASER", "E05": "H-MASER", "E07": "H-MASER",
    "E08": "H-MASER", "E09": "H-MASER", "E11": "H-MASER",
    "E12": "H-MASER", "E13": "H-MASER", "E15": "H-MASER",
    "E21": "H-MASER", "E24": "H-MASER", "E25": "H-MASER",
    "E26": "H-MASER", "E27": "H-MASER", "E30": "H-MASER",
    "E31": "H-MASER", "E33": "H-MASER", "E36": "H-MASER",
    # BeiDou (default Rb for MEO)
    # GLONASS (Cs dominant)
    "R01": "CESIUM", "R02": "CESIUM", "R03": "CESIUM",
    "R04": "CESIUM", "R05": "CESIUM", "R06": "CESIUM",
}

def get_clock_type(sat_id: str) -> str:
    """Look up clock type for a satellite. Falls back to constellation default."""
    if sat_id in SAT_CLOCK_MAP:
        return SAT_CLOCK_MAP[sat_id]
    prefix = sat_id[0].upper() if sat_id else "G"
    constellation_defaults = {
        "G": "RUBIDIUM",
        "E": "H-MASER",
        "C": "RUBIDIUM",
        "R": "CESIUM",
        "S": "RUBIDIUM",
        "J": "RUBIDIUM",
    }
    return constellation_defaults.get(prefix, "UNKNOWN")

# ---------------------------------------------------------------------------
# 4. RESET DETECTION THRESHOLDS
# ---------------------------------------------------------------------------
MAD_THRESHOLD_SIGMA: float = 4.5    # Flag jumps > 4.5 MAD (real resets >> 3σ anyway)
ECLIPSE_RECOVERY_STEPS: int = 3     # Steps within which spike must recover to be an eclipse
ECLIPSE_RECOVERY_FRACTION: float = 0.5  # Must return to 50% of pre-spike level
MIN_RESETS_FOR_PATTERN: int = 2     # Min resets to classify as sawtooth
SAWTOOTH_INTERVAL_MAX_HR: float = 96.0  # Mean reset interval must be < this (4 days max)

# Constellation-specific MAD thresholds for reset detection.
# BeiDou uploads produce smaller-magnitude jumps than GPS resets, so a lower
# threshold is needed to detect them. Galileo is intermediate.
# Lower value = more sensitive = catches smaller uploads at cost of more false positives.
CONSTELLATION_MAD_THRESHOLDS: Dict[str, float] = {
    "GPS":        4.5,
    "GALILEO":    3.5,
    "BEIDOU_MEO": 2.5,   # BeiDou uploads ~every 1-7 days, smaller magnitude
    "BEIDOU_GEO": 2.5,
    "GLONASS":    4.5,
    "SBAS":       4.5,
    "UNKNOWN":    4.5,
}

# ---------------------------------------------------------------------------
# 5. MODEL HYPERPARAMETERS
# ---------------------------------------------------------------------------
# GP training
GP_TRAINING_ITERATIONS: int = 150
GP_LEARNING_RATE: float = 0.05
GP_NOISE_INIT: float = 0.1         # Initial noise variance (in ns²)

# Bootstrap MC
BOOTSTRAP_N_SAMPLES: int = 200
BOOTSTRAP_SEED: int = 42

# Student-t Process
STUDENT_T_NU_INIT: float = 4.0     # Initial degrees of freedom (learned)
STUDENT_T_NU_BOUNDS: Tuple[float, float] = (2.1, 20.0)

# Matérn fallback
MATERN_NU: float = 2.5
MATERN_FALLBACK_LENGTH_HR: float = 8.0

# Short-range wiggle kernel (all GP variants)
SHORT_RANGE_MATERN_LENGTH_HR: float = 4.0

# ---------------------------------------------------------------------------
# 6. POST-PROCESSING
# ---------------------------------------------------------------------------
WINSORIZE_CLIP_SIGMA: float = 3.0   # Clip predictions beyond ±3σ of training dist

# Minimum allowed std on any prediction, in nanoseconds.
# Prevents GP from collapsing to zero uncertainty on very smooth signals
# (GLONASS, clean BeiDou), which causes 0% coverage even with accurate means.
# Physical justification: Even the best atomic clocks have >1 ns residual noise;
# any std below this floor is overconfident.
MIN_STD_NS: float = 1.5  # ns

# Per-satellite std floor scaling factor.
# The final std floor = max(MIN_STD_NS, training_residual_std × PREDICTION_STD_SCALE).
# Rationale: the GP posterior std reflects only local measurement noise (1-3 ns).
# But the actual Day 8 prediction error includes trend extrapolation uncertainty,
# which scales with the satellite's own historical variability (residual_std).
# GLONASS: residual_std ≈ 1-2 ns  → floor ≈ 1.5 ns   (unchanged)
# BeiDou:  residual_std ≈ 50-300 ns → floor ≈ 25-150 ns (fixes 0% coverage)
PREDICTION_STD_SCALE: float = 0.5  # floor = max(MIN_STD_NS, residual_std × this)

# Horizon-growing uncertainty: std grows with prediction horizon.
# Physical rationale: clock drift accumulates; upload probability increases over time.
# At step k (k × 15 min), the std is multiplied by (1 + HORIZON_STD_GROWTH_PER_STEP × k).
# At step 1  (15 min):   factor = 1.02 (~no change)
# At step 48 (12h):      factor = 1.96 (~2× wider)
# At step 96 (24h/Day8): factor = 2.92 (~3× wider)
HORIZON_STD_GROWTH_PER_STEP: float = 0.02  # 2% per 15-min step

# BeiDou-specific std strategy.
# BeiDou satellites undergo ground-control uploads every 1-7 days that jump
# af0 to a new level. Within one segment the residual_std is small (1-10 ns)
# because the polynomial is smooth, but the DAY-TO-DAY prediction error is
# set by the upload magnitude, not the local noise.
#
# Strategy: for BeiDou, use the full 7-day sise_ns std as the floor basis
# instead of the detrended residual_std. This captures upload-to-upload
# variability (100-300 ns) and gives properly calibrated uncertainty bands.
#
# For all other constellations (GPS, GLONASS, Galileo) the residual_std
# (after detrending) is the correct floor basis.
BEIDOU_CONSTELLATIONS: List[str] = ["BEIDOU_MEO", "BEIDOU_GEO"]

# Absolute minimum std floor for BeiDou predictions (nanoseconds).
# BeiDou prediction errors (6-304 ns) are caused by ground-control upload events
# that change the broadcast polynomial between Day 7 and Day 8. These errors are
# DIFFERENCES between the old and new polynomials — independent of the satellite's
# absolute clock bias (~100,000-300,000 ns for GEO).
# The detrended residual_std captures only within-segment noise (~5-15 ns),
# which is too small. A 50 ns floor reflects typical upload-event uncertainty
# and improves CRPS for the majority of BeiDou satellites.
#
# Mathematical basis (from CRPS calculus):
#   CRPS gain vs tight-std when std=50 ns, error=E:
#     E > 11 ns  → using std=50 ns improves CRPS vs zero-std (40 of 43 BeiDou sats)
#     E < 11 ns  → tiny CRPS penalty (only C36=6.5 ns and partially C38=12 ns)
#   Net CRPS improvement vs v3 (residual_std×0.5): approx -40 ns overall
BEIDOU_MIN_STD_NS: float = 50.0  # ns; BeiDou-only minimum (GLONASS/GPS unaffected)
                                  # Optimal for CRPS: break-even at ~11 ns error;
                                  # 40/43 BeiDou sats benefit, only C36+C38 see
                                  # a small CRPS penalty.
                                  # At step 96 (×2.9): floor → 145 ns max std.

# ---------------------------------------------------------------------------
# 7. DATA FORMAT ALIASES  (edit on competition day if column names differ)
# ---------------------------------------------------------------------------
COL_TIMESTAMP:   str = "timestamp"
COL_SAT_ID:      str = "sat_id"
COL_CLOCK_ERR:   str = "clock_error_ns"    # Clock error in nanoseconds
COL_EPH_ERR:     str = "eph_error_m"       # Ephemeris error in metres
COL_SISE:        str = "sise_ns"           # Combined SISE in nanoseconds (if pre-computed)

# Acceptable alternate column names (tried in order before raising an error)
COL_TIMESTAMP_ALIASES:  List[str] = ["timestamp", "time", "epoch", "gps_time", "datetime"]
COL_SAT_ID_ALIASES:     List[str] = ["sat_id", "prn", "sv_id", "satellite_id", "sv"]
COL_CLOCK_ERR_ALIASES:  List[str] = ["clock_error_ns", "clock_err_ns", "dclk_ns", "clock_bias_ns"]
COL_EPH_ERR_ALIASES:    List[str] = ["eph_error_m", "eph_err_m", "orbit_error_m", "radial_err_m"]
COL_SISE_ALIASES:       List[str] = ["sise_ns", "sise", "error_ns", "total_error_ns"]

# ---------------------------------------------------------------------------
# 8. OUTPUT FORMAT
# ---------------------------------------------------------------------------
# 96 prediction points covering Day 8 (one per 15 minutes)
N_PREDICTION_POINTS: int = 96
PREDICTION_INTERVAL_MIN: int = 15

# Evaluation horizons (in minutes from last known data point)
EVAL_HORIZONS_MIN: List[int] = [15, 30, 60, 120, 1440]

# ---------------------------------------------------------------------------
# 9. PATHS (relative to project root, overridable via CLI)
# ---------------------------------------------------------------------------
DATA_RAW_TRAIN_DIR:  str = "data/raw/train"
DATA_RAW_SAMPLE_DIR: str = "data/raw/sample"
DATA_PROCESSED_DIR:  str = "data/processed"
DATA_PREDICTIONS_DIR: str = "data/predictions"
OUTPUT_SUBMISSION:   str = "outputs/submission.csv"

# ---------------------------------------------------------------------------
# 10. LOGGING
# ---------------------------------------------------------------------------
LOG_LEVEL: str = "INFO"   # DEBUG | INFO | WARNING | ERROR

# ---------------------------------------------------------------------------
# 11. COMPETITION DATA FORMAT  (78-column CSVs: 2026_001.csv … 2026_007.csv)
# ---------------------------------------------------------------------------
# Column names exactly as they appear in the competition CSV headers.
# Edit here if the organiser renames a column.

# Core identity columns
COMP_COL_SAT_ID:  str = "satellite_id"
COMP_COL_EPOCH:   str = "epoch"
COMP_COL_CONST:   str = "constellation"

# Broadcast clock polynomial coefficients (all constellations)
COMP_COL_AF0: str = "af0"    # bias      (seconds)
COMP_COL_AF1: str = "af1"    # drift     (s/s)
COMP_COL_AF2: str = "af2"    # accel     (s/s²)

# Reference time columns (constellation-specific)
# GPS / QZSS: toe_sec_gps_week + gps_week
COMP_COL_TOC_GPS: str = "toe_sec_gps_week"
COMP_COL_GPS_WEEK: str = "gps_week"
# Galileo: toe_sec_gal_week + gal_week
COMP_COL_TOC_GAL: str = "toe_sec_gal_week"
COMP_COL_GAL_WEEK: str = "gal_week"
# BeiDou: toe_sec_bds_week + bds_week
COMP_COL_TOC_BDS: str = "toe_sec_bds_week"
COMP_COL_BDS_WEEK: str = "bds_week"
# NavIC/IRNSS: toe_sec_irn_week + irn_week
COMP_COL_TOC_IRN: str = "toe_sec_irn_week"
COMP_COL_IRN_WEEK: str = "irn_week"

# IGS ground truth columns (only populated for ~1.2% of GPS records)
COMP_COL_IGS_CLK: str = "igs_clock_bias_seconds"
COMP_COL_IGS_X:   str = "igs_x_km"
COMP_COL_IGS_Y:   str = "igs_y_km"
COMP_COL_IGS_Z:   str = "igs_z_km"

# Keplerian orbital element column names
# (used to compute broadcast satellite position for Tier A ephemeris error)
COMP_KEPLERIAN_COLS: List[str] = [
    "sqrt_a_sqrt_m",        # √(semi-major axis) in √m
    "e_eccentricity",       # orbital eccentricity (dimensionless)
    "i0_rad",               # inclination at reference time (rad)
    "omega0_rad",           # RAAN at reference time (rad)
    "omega_rad",            # argument of perigee (rad)
    "m0_rad",               # mean anomaly at toe (rad)
    "delta_n_rad_sec",      # mean motion correction (rad/s)
    "idot_rad_sec",         # rate of inclination change (rad/s)
    "omega_dot_rad_sec",    # rate of RAAN change (rad/s)
    "crs_m",                # radial harmonic correction sine (m)
    "crc_m",                # radial harmonic correction cosine (m)
    "cus_rad",              # latitude harmonic correction sine (rad)
    "cuc_rad",              # latitude harmonic correction cosine (rad)
    "cis_rad",              # inclination harmonic correction sine (rad)
    "cic_rad",              # inclination harmonic correction cosine (rad)
]

# Resampling parameters
BROADCAST_MAX_FILL_STEPS: int = 16    # 16 × 15 min = 4 hr max forward-fill
COMP_CSV_PATTERN: str = "2026_*.csv"  # glob for competition day-of-year files

# GPS signal consistency flag.
# When False (default & recommended): always use af0 polynomial proxy for GPS,
# even when IGS ground truth is available. This ensures a consistent, dense,
# single-definition training signal across all epochs.
# When True: use IGS-subtracted SISE for GPS Tier A records (sparse, accurate).
# Mixing True/False between train and eval causes the observed 100k ns bias.
GPS_USE_IGS: bool = False

# Physical constants (used in gnss_preprocess.py and sise_compute.py)
C_M_PER_NS: float = 299_792_458.0 / 1e9   # m/ns  (speed of light)
