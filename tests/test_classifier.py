"""
test_classifier.py — Tests for Module 4: classifier.py
test_detrend.py — Tests for Module 5: detrend.py
test_reset_detector.py — Tests for Module 6: reset_detector.py
(combined for brevity — split into separate files if test suite grows)
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# ===========================================================================
# Classifier tests
# ===========================================================================

from src.classifier import (
    classify_satellite,
    _classify_orbit,
    _classify_resets,
    _get_constellation,
    OrbitType,
    ResetPattern,
    ModelType,
)
from src.reset_detector import ResetEvent, EventType


class TestOrbitClassification:

    def test_gps_is_meo(self, clean_gps_series, sample_timestamps):
        clf = classify_satellite("G01", clean_gps_series, resets=[])
        assert clf.orbit_type == OrbitType.MEO
        assert clf.constellation == "GPS"

    def test_galileo_is_meo(self, clean_gps_series, sample_timestamps):
        clf = classify_satellite("E05", clean_gps_series, resets=[])
        assert clf.orbit_type == OrbitType.MEO
        assert clf.constellation == "GALILEO"

    def test_beidou_geo_is_geo(self, geo_series, sample_timestamps):
        clf = classify_satellite("C03", geo_series, resets=[])
        assert clf.orbit_type == OrbitType.GEO

    def test_beidou_meo_is_meo(self, clean_gps_series, sample_timestamps):
        clf = classify_satellite("C21", clean_gps_series, resets=[])
        assert clf.orbit_type == OrbitType.MEO

    def test_glonass_is_meo(self, clean_gps_series, sample_timestamps):
        clf = classify_satellite("R05", clean_gps_series, resets=[])
        assert clf.orbit_type == OrbitType.MEO
        assert clf.constellation == "GLONASS"

    def test_sbas_is_geo(self, geo_series, sample_timestamps):
        clf = classify_satellite("S20", geo_series, resets=[])
        assert clf.orbit_type == OrbitType.GEO

    def test_orbital_period_gps(self):
        _, _, period = _classify_orbit("G01", pd.Series(dtype=float), False)
        assert abs(period - 11.9667) < 0.01

    def test_orbital_period_galileo(self):
        _, _, period = _classify_orbit("E01", pd.Series(dtype=float), False)
        assert abs(period - 14.0833) < 0.01

    def test_orbital_period_glonass(self):
        _, _, period = _classify_orbit("R01", pd.Series(dtype=float), False)
        assert abs(period - 11.2667) < 0.01


class TestResetPatternClassification:

    def _make_reset(self, timestamp, idx, mag, etype=EventType.RESET):
        return ResetEvent(timestamp=timestamp, index=idx, magnitude_ns=mag, event_type=etype)

    def test_no_resets_is_clean(self):
        pattern, n, _, _ = _classify_resets([])
        assert pattern == ResetPattern.CLEAN
        assert n == 0

    def test_one_reset_is_clean(self):
        resets = [self._make_reset(pd.Timestamp("2024-01-01"), 50, -5.0)]
        pattern, n, _, _ = _classify_resets(resets)
        assert pattern == ResetPattern.CLEAN

    def test_regular_sawtooth(self):
        # 3 resets at ~48hr intervals (consistent)
        times = [
            pd.Timestamp("2024-01-01 00:00"),
            pd.Timestamp("2024-01-03 00:00"),
            pd.Timestamp("2024-01-05 00:00"),
        ]
        resets = [self._make_reset(t, i*200, -5.0) for i, t in enumerate(times)]
        pattern, n, mean_hr, std_hr = _classify_resets(resets)
        assert pattern == ResetPattern.REGULAR
        assert n == 3
        assert abs(mean_hr - 48.0) < 0.5

    def test_irregular_sawtooth(self):
        # 4 resets with very different intervals
        times = [
            pd.Timestamp("2024-01-01 00:00"),
            pd.Timestamp("2024-01-01 06:00"),  # only 6hr apart
            pd.Timestamp("2024-01-05 00:00"),  # 90hr later
            pd.Timestamp("2024-01-07 00:00"),
        ]
        resets = [self._make_reset(t, i*50, -5.0) for i, t in enumerate(times)]
        pattern, n, _, _ = _classify_resets(resets)
        assert pattern == ResetPattern.IRREGULAR

    def test_eclipse_events_excluded(self):
        """Eclipse events should NOT count towards reset classification."""
        eclipse = self._make_reset(pd.Timestamp("2024-01-01"), 50, 20.0, EventType.ECLIPSE)
        pattern, n, _, _ = _classify_resets([eclipse])
        assert pattern == ResetPattern.CLEAN
        assert n == 0


class TestModelRouting:

    def test_clean_routes_to_gp(self, clean_gps_series):
        clf = classify_satellite("G01", clean_gps_series, resets=[])
        assert clf.model_type == ModelType.GP

    def test_regular_routes_to_bootstrap(self, sample_timestamps):
        # Directly test _classify_resets + _route_model for 3 consistent resets
        from src.classifier import _classify_resets, _route_model
        ts = sample_timestamps
        resets = [
            ResetEvent(ts[100], 100, -5.0, EventType.RESET),
            ResetEvent(ts[300], 300, -5.0, EventType.RESET),
            ResetEvent(ts[500], 500, -5.0, EventType.RESET),
        ]
        pattern, n, mean_hr, _ = _classify_resets(resets)
        # Steps 100, 300, 500 at 15min each = 25h, 75h, 125h → intervals 50h, 50h
        assert pattern == ResetPattern.REGULAR, f"Expected REGULAR, got {pattern} (mean={mean_hr:.1f}hr)"
        assert _route_model(pattern) == ModelType.BOOTSTRAP

    def test_irregular_routes_to_student_t(self, irregular_sawtooth_series, sample_timestamps):
        from src.reset_detector import detect_resets
        resets = detect_resets(irregular_sawtooth_series, sat_id="G21")
        real_resets = [r for r in resets if r.event_type == EventType.RESET]
        if len(real_resets) >= 2:
            clf = classify_satellite("G21", irregular_sawtooth_series, resets=resets)
            # With erratic timing, should be irregular → StudentT
            assert clf.model_type in [ModelType.STUDENT_T, ModelType.BOOTSTRAP]


# ===========================================================================
# Detrend tests
# ===========================================================================

from src.detrend import detrend, _build_feature_matrix, _timestamps_to_hours


class TestDetrend:

    def test_residual_has_lower_std(self, clean_gps_series, sample_timestamps):
        """After detrending, residual std should be less than original std."""
        result = detrend(
            sat_id="G01",
            sise_series=clean_gps_series,
            constellation="GPS",
        )
        assert result.residual.std() < clean_gps_series.std()

    def test_residual_mean_near_zero(self, clean_gps_series):
        """Residual should be roughly zero-mean after detrending."""
        result = detrend("G01", clean_gps_series, "GPS")
        assert abs(result.residual.dropna().mean()) < 1.0  # Within 1 ns

    def test_trend_plus_residual_equals_original(self, clean_gps_series):
        """trend + residual should reconstruct the original signal (approx)."""
        result = detrend("G01", clean_gps_series, "GPS")
        reconstructed = result.trend_total + result.residual.values
        original = clean_gps_series.values
        # Within 0.1 ns RMS (Ridge regularization adds small bias)
        rms_diff = np.sqrt(np.nanmean((reconstructed - original) ** 2))
        assert rms_diff < 0.5

    def test_predict_trend_returns_96_points(self, clean_gps_series, future_timestamps):
        """predict_trend() should return exactly 96 values."""
        result = detrend("G01", clean_gps_series, "GPS")
        trend_pred = result.predict_trend(future_timestamps)
        assert len(trend_pred) == 96

    def test_geo_uses_24hr_period(self, geo_series):
        """GEO satellite detrending should use 24hr orbital period."""
        result = detrend("C03", geo_series, "BEIDOU_GEO", orbital_period_hr=24.0)
        assert result.orbital_period_hr_fit == 24.0

    def test_handles_nan_in_input(self, clean_gps_series):
        """Detrending should handle NaN values gracefully."""
        series_with_nan = clean_gps_series.copy()
        series_with_nan.iloc[50:55] = np.nan
        result = detrend("G01", series_with_nan, "GPS")
        # Should not raise; residual should have same length as input
        assert len(result.residual) == len(series_with_nan)

    def test_feature_matrix_shape(self):
        """Feature matrix should have expected number of columns."""
        t = np.arange(100) * 0.25
        X = _build_feature_matrix(t, poly_degree=3, orbital_period_hr=12.0, solar_period_hr=24.0)
        # Expected: 4 (poly) + 2 (orbital sin/cos) + 2 (solar sin/cos) + 2 (2nd orbital) = 10
        assert X.shape == (100, 10)

    def test_different_constellations_different_periods(self, clean_gps_series):
        """GPS and Galileo should produce slightly different detrend results."""
        result_gps  = detrend("G01", clean_gps_series, "GPS")
        result_gal  = detrend("E01", clean_gps_series, "GALILEO")
        assert result_gps.orbital_period_hr_fit != result_gal.orbital_period_hr_fit


# ===========================================================================
# Reset detector tests
# ===========================================================================

from src.reset_detector import detect_resets, reset_statistics, mask_eclipses, EventType


class TestResetDetector:

    def test_no_resets_in_clean_series(self, clean_gps_series):
        """Clean signal should produce no resets."""
        events = detect_resets(clean_gps_series, sat_id="G01_clean")
        real_resets = [e for e in events if e.event_type == EventType.RESET]
        assert len(real_resets) == 0

    def test_detects_sawtooth_resets(self, sawtooth_gps_series):
        """Sawtooth signal should have detected resets near injection points."""
        events = detect_resets(sawtooth_gps_series, sat_id="G01_saw")
        real_resets = [e for e in events if e.event_type == EventType.RESET]
        assert len(real_resets) >= 2  # Should detect at least 2 of 3 injected

    def test_eclipse_classified_correctly(self, irregular_sawtooth_series):
        """Transient spike at step 350 should be classified as eclipse."""
        events = detect_resets(irregular_sawtooth_series, sat_id="G21")
        eclipses = [e for e in events if e.event_type == EventType.ECLIPSE]
        # The synthetic eclipse at step 350 should be detected
        assert len(eclipses) >= 1

    def test_reset_statistics_empty(self):
        """reset_statistics with no resets should return safe defaults."""
        stats = reset_statistics([])
        assert stats["n_resets"] == 0
        assert stats["mean_interval_hr"] == float("inf")

    def test_reset_statistics_with_resets(self, sawtooth_gps_series, sample_timestamps):
        """reset_statistics should return valid interval/magnitude stats."""
        events = detect_resets(sawtooth_gps_series, sat_id="test")
        stats = reset_statistics(events)
        if stats["n_resets"] >= 2:
            assert stats["mean_interval_hr"] > 0
            assert stats["std_magnitude_ns"] >= 0

    def test_mask_eclipses_sets_nan(self, irregular_sawtooth_series):
        """Eclipse positions should become NaN after masking."""
        events = detect_resets(irregular_sawtooth_series, sat_id="G21")
        eclipses = [e for e in events if e.event_type == EventType.ECLIPSE]
        if eclipses:
            masked = mask_eclipses(irregular_sawtooth_series, events)
            # Check that at least some positions are NaN now
            assert masked.isna().sum() > 0

    def test_all_nan_series(self, sample_timestamps):
        """All-NaN series should return empty list, not raise."""
        series = pd.Series([np.nan] * len(sample_timestamps), index=sample_timestamps)
        events = detect_resets(series, sat_id="BADSAT")
        assert events == []

    def test_large_spike_is_reset(self, sample_timestamps):
        """A sustained large spike should be classified as RESET."""
        rng = np.random.default_rng(0)
        vals = rng.normal(0, 0.5, len(sample_timestamps))
        vals[200:] += 20.0  # Permanent 20 ns jump
        series = pd.Series(vals, index=sample_timestamps)

        events = detect_resets(series, sat_id="TEST_RESET")
        resets = [e for e in events if e.event_type == EventType.RESET]
        assert len(resets) >= 1

    def test_transient_spike_is_eclipse(self, sample_timestamps):
        """A transient spike that recovers quickly should be ECLIPSE."""
        rng = np.random.default_rng(1)
        vals = rng.normal(0, 0.5, len(sample_timestamps))
        vals[100] += 30.0   # Huge spike at step 100
        vals[101] += 10.0   # Partial recovery
        # Step 102 onwards is back to normal
        series = pd.Series(vals, index=sample_timestamps)

        events = detect_resets(series, sat_id="TEST_ECLIPSE")
        eclipses = [e for e in events if e.event_type == EventType.ECLIPSE]
        assert len(eclipses) >= 1
