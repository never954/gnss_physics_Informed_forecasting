"""Outlier treatment — the dominant lever for the Shapiro-Wilk objective.

The error signal is a clean Gaussian core plus sparse heavy tails. Clipping the
training points at a MAD-scaled threshold stops any model from chasing those
spikes, so the fit tracks the central process and does not inject extra
non-Gaussian structure into the residual.
"""
import numpy as np


def mad_clip(y, thresh=3.5):
    """Clip `y` to [median +/- thresh * 1.4826 * MAD]. Returns clipped copy."""
    y = np.asarray(y, dtype=float)
    med = np.median(y)
    mad = np.median(np.abs(y - med)) * 1.4826 + 1e-12
    return np.clip(y, med - thresh * mad, med + thresh * mad)


def outlier_mask(y, thresh=3.5):
    """Boolean mask: True where `y` is within thresh MAD of the median (an inlier)."""
    y = np.asarray(y, dtype=float)
    med = np.median(y)
    mad = np.median(np.abs(y - med)) * 1.4826 + 1e-12
    return np.abs(y - med) <= thresh * mad
