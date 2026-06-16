"""
conftest.py — Shared pytest fixtures for all test modules.

Generates synthetic GNSS SISE-like data for testing each module
without requiring real competition data.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_timestamps():
    """7 days of 15-minute timestamps (672 points)."""
    return pd.date_range("2024-01-01", periods=672, freq="15min", tz="UTC")


@pytest.fixture
def clean_gps_series(sample_timestamps):
    """
    Synthetic clean GPS satellite signal:
    - Slow drift (linear)
    - Orbital harmonic at 11.97hr
    - Solar harmonic at 24hr
    - Low-amplitude Gaussian noise
    """
    rng = np.random.default_rng(42)
    t = np.arange(len(sample_timestamps)) * 0.25  # hours

    drift     = 0.01 * t                                           # 0.01 ns/hr drift
    orbital   = 2.0 * np.sin(2 * np.pi * t / 11.9667)            # 2 ns orbital wave
    solar     = 1.0 * np.cos(2 * np.pi * t / 24.0)               # 1 ns solar wave
    noise     = rng.normal(0, 0.3, len(t))                        # 0.3 ns noise

    sise = drift + orbital + solar + noise
    return pd.Series(sise, index=sample_timestamps, name="sise_ns")


@pytest.fixture
def sawtooth_gps_series(sample_timestamps):
    """
    Synthetic regular sawtooth GPS satellite:
    Same as clean but with 3 ground-upload resets (every ~56 hours).
    """
    rng = np.random.default_rng(99)
    t = np.arange(len(sample_timestamps)) * 0.25

    drift   = 0.01 * t
    orbital = 2.0 * np.sin(2 * np.pi * t / 11.9667)
    solar   = 1.0 * np.cos(2 * np.pi * t / 24.0)
    noise   = rng.normal(0, 0.3, len(t))

    sise = drift + orbital + solar + noise

    # Add 3 resets at steps 100, 300, 500
    for reset_idx in [100, 300, 500]:
        sise[reset_idx:] += rng.normal(-5.0, 0.5)  # ~5 ns reset magnitude

    return pd.Series(sise, index=sample_timestamps, name="sise_ns")


@pytest.fixture
def irregular_sawtooth_series(sample_timestamps):
    """
    Synthetic irregular sawtooth (like G21):
    Erratic reset timing + 2 outlier spikes.
    """
    rng = np.random.default_rng(77)
    t = np.arange(len(sample_timestamps)) * 0.25

    drift   = 0.015 * t
    orbital = 3.0 * np.sin(2 * np.pi * t / 11.9667)
    solar   = 1.5 * np.cos(2 * np.pi * t / 24.0)
    noise   = rng.normal(0, 0.5, len(t))

    sise = drift + orbital + solar + noise

    # Erratic resets
    for reset_idx, mag in [(80, -8.0), (200, 12.0), (450, -6.0), (580, 15.0)]:
        sise[reset_idx:] += rng.normal(mag, 2.0)

    # Eclipse spike (transient, recovers in 2 steps)
    sise[350] += 20.0
    sise[351] += 10.0
    sise[352] += 2.0

    return pd.Series(sise, index=sample_timestamps, name="sise_ns")


@pytest.fixture
def geo_series(sample_timestamps):
    """
    Synthetic GEO satellite:
    Dominant 24hr harmonic, small orbital component (same as solar for GEO).
    """
    rng = np.random.default_rng(55)
    t = np.arange(len(sample_timestamps)) * 0.25

    drift  = 0.005 * t
    diurnal = 3.0 * np.sin(2 * np.pi * t / 24.0 + 0.5)
    noise  = rng.normal(0, 0.2, len(t))

    sise = drift + diurnal + noise
    return pd.Series(sise, index=sample_timestamps, name="sise_ns")


@pytest.fixture
def future_timestamps(sample_timestamps):
    """96 future timestamps (Day 8) at 15-min cadence."""
    last = sample_timestamps[-1]
    return pd.date_range(last + pd.Timedelta("15min"), periods=96, freq="15min", tz="UTC")
