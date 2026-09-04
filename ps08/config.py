"""Central configuration for the PS-08 GNSS error-prediction pipeline.

Everything data-format-specific lives here so the rest of the code stays generic.
"""
import os

# Data location, resolved in order:
#   1. $PS08_DATA_DIR  (explicit override)
#   2. ./data next to this file  (self-contained repo copy)
#   3. the original absolute path  (local dev fallback)
_HERE = os.path.dirname(os.path.abspath(__file__))
_LOCAL_DATA = os.path.join(_HERE, "data")
DATA_DIR = (
    os.environ.get("PS08_DATA_DIR")
    or (_LOCAL_DATA if os.path.isdir(_LOCAL_DATA) else "/Users/vedantghule/Downloads/Data_PS-08")
)

# Canonical channel names AFTER whitespace normalisation (collapse internal spaces).
# The raw files have quirks like "y_error  (m)" with a double space — loader fixes that.
CHANNELS = ["x_error (m)", "y_error (m)", "z_error (m)", "satclockerror (m)"]

TIME_COL = "utc_time"
TIME_FMT = "%m/%d/%Y %H:%M"

# Train -> Test pairs. period_h = dominant physical period hint (hours):
#   GEO ~ sidereal day (23.93 h),  MEO ~ half-sidereal orbital cycle (11.97 h).
# It is only a *prior*; models with a learnable period optimise around it.
PAIRS = [
    # label, train file,            test file,             orbit, period_h
    ("GEO",  "DATA_GEO_Train.csv",  "DATA_GEO_Test.csv",   "GEO", 23.93),
    ("MEO1", "DATA_MEO_Train.csv",  "DATA_MEO_Test.csv",   "MEO", 11.97),
    ("MEO2", "DATA_MEO_Train2.csv", "DATA_MEO_Test2.csv",  "MEO", 11.97),
]

# Outlier treatment: how many robust sigma (MAD-scaled) before a training point is clipped.
MAD_CLIP_THRESH = 3.5

RANDOM_SEED = 42
