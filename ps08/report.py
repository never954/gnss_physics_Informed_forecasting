"""Validated Gaussian reporting.

Two jobs:
  1. validate_sw() — confirm our Shapiro-Wilk implementation matches the evaluator's by
     scoring the provided benchmark sample (SW_ReferenceData.xlsx, expected W ~ 0.985).
  2. shapiro_row() — the per-channel residual normality line used in the reports.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import shapiro

from config import DATA_DIR

SW_REFERENCE = os.path.join(DATA_DIR, "SW_ReferenceData.xlsx")
ALPHA = 0.05  # significance level from the problem statement


def validate_sw():
    """Score the benchmark reference sample. Returns (W, p, n)."""
    vals = pd.read_excel(SW_REFERENCE, header=None).iloc[:, 0].astype(float).to_numpy()
    W, p = shapiro(vals)
    return float(W), float(p), len(vals)


def shapiro_row(residual):
    """Return (W, p, H0_rejected, mean, std, n) for a residual array."""
    r = np.asarray(residual, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 3:
        return (np.nan, np.nan, None, np.nan, np.nan, len(r))
    W, p = shapiro(r)
    rejected = bool(p < ALPHA)  # True => residual is NOT normal (bad)
    return (float(W), float(p), rejected, float(np.mean(r)), float(np.std(r)), len(r))
