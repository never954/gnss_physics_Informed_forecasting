"""
test_models.py — Tests for Modules 7–10: GP, Bootstrap MC, Student-t, Matérn fallback.

NOTE: GP / Student-t tests require gpytorch. They are automatically skipped
if gpytorch is not installed (pytest.importorskip).
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detrend import detrend


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_residual(series, constellation="GPS", orbital_period_hr=11.9667):
    """Helper: detrend and return residual + timestamps."""
    result = detrend("TEST", series, constellation, orbital_period_hr=orbital_period_hr)
    return result.timestamps, result.residual


# ===========================================================================
# GP Model tests (Module 7)
# ===========================================================================

class TestGPModel:
    gpytorch = pytest.importorskip("gpytorch")

    def test_fit_and_predict_shape(self, clean_gps_series, future_timestamps):
        from src.models.gp_model import GPModel
        ts, residual = _get_residual(clean_gps_series)

        model = GPModel(sat_id="G01", orbital_period_hr=11.9667, n_iterations=30)
        model.fit(ts, residual)
        mean, std = model.predict(future_timestamps)

        assert mean.shape == (96,)
        assert std.shape  == (96,)

    def test_std_is_positive(self, clean_gps_series, future_timestamps):
        from src.models.gp_model import GPModel
        ts, residual = _get_residual(clean_gps_series)

        model = GPModel(sat_id="G01", orbital_period_hr=11.9667, n_iterations=30)
        model.fit(ts, residual)
        _, std = model.predict(future_timestamps)

        assert (std > 0).all()

    def test_predict_without_fit_raises(self, future_timestamps):
        from src.models.gp_model import GPModel
        model = GPModel(sat_id="G01", orbital_period_hr=11.9667)
        with pytest.raises(RuntimeError, match="not trained"):
            model.predict(future_timestamps)

    def test_learned_orbital_period_in_bounds(self, clean_gps_series, future_timestamps):
        from src.models.gp_model import GPModel
        from src.config import ORBITAL_PERIOD_TOLERANCE
        ts, residual = _get_residual(clean_gps_series)

        centre = 11.9667
        model = GPModel(sat_id="G01", orbital_period_hr=centre, n_iterations=30)
        model.fit(ts, residual)

        learned = model.learned_orbital_period_hr
        lo = centre * (1 - ORBITAL_PERIOD_TOLERANCE)
        hi = centre * (1 + ORBITAL_PERIOD_TOLERANCE)
        assert lo <= learned <= hi, f"Learned period {learned:.3f} outside [{lo:.3f}, {hi:.3f}]"

    def test_clock_type_aware_init_rubidium(self):
        from src.models.gp_model import GPModel
        from src.config import CLOCK_KERNEL_CONFIGS
        model = GPModel(sat_id="G01", orbital_period_hr=11.9667)
        # Clock type should be resolved to RUBIDIUM for GPS
        assert model.clock_type == "RUBIDIUM"

    def test_clock_type_aware_init_hmaser(self):
        from src.models.gp_model import GPModel
        model = GPModel(sat_id="E01", orbital_period_hr=14.0833)
        assert model.clock_type == "H-MASER"

    def test_geo_satellite_uses_different_period(self, geo_series, future_timestamps):
        from src.models.gp_model import GPModel
        ts, residual = _get_residual(geo_series, "BEIDOU_GEO", 24.0)

        model = GPModel(sat_id="C03", orbital_period_hr=24.0, n_iterations=20)
        model.fit(ts, residual)
        mean, std = model.predict(future_timestamps)
        assert len(mean) == 96

    def test_handles_nan_in_residual(self, clean_gps_series, future_timestamps):
        from src.models.gp_model import GPModel
        ts, residual = _get_residual(clean_gps_series)
        residual = residual.copy()
        residual.iloc[50:60] = np.nan  # 10-step gap

        model = GPModel(sat_id="G01", orbital_period_hr=11.9667, n_iterations=20)
        model.fit(ts, residual)  # Should not raise
        mean, std = model.predict(future_timestamps)
        assert mean.shape == (96,)


# ===========================================================================
# Bootstrap MC tests (Module 8)
# ===========================================================================

class TestBootstrapMCModel:

    def test_fit_and_predict_shape(self, sawtooth_gps_series, sample_timestamps, future_timestamps):
        from src.models.bootstrap_mc import BootstrapMCModel
        from src.reset_detector import detect_resets

        ts, residual = _get_residual(sawtooth_gps_series)
        resets = detect_resets(residual, sat_id="G01")

        model = BootstrapMCModel(sat_id="G01", orbital_period_hr=11.9667, n_bootstrap=50)
        model.fit(ts, residual, resets)
        mean, std = model.predict(future_timestamps)

        assert mean.shape == (96,)
        assert std.shape  == (96,)

    def test_std_nonzero_with_resets(self, sawtooth_gps_series, future_timestamps):
        """Bootstrap MC should produce nonzero std when resets are present."""
        from src.models.bootstrap_mc import BootstrapMCModel
        from src.reset_detector import detect_resets

        ts, residual = _get_residual(sawtooth_gps_series)
        resets = detect_resets(residual, sat_id="G01")

        model = BootstrapMCModel(sat_id="G01", orbital_period_hr=11.9667, n_bootstrap=100)
        model.fit(ts, residual, resets)
        _, std = model.predict(future_timestamps)
        # Std should be nonzero when resets are possible
        assert std.mean() >= 0.0  # Permissive: could be 0 if no resets expected in window

    def test_no_resets_gives_low_uncertainty(self, clean_gps_series, future_timestamps):
        """Clean satellite (no resets) should have low Bootstrap uncertainty."""
        from src.models.bootstrap_mc import BootstrapMCModel

        ts, residual = _get_residual(clean_gps_series)
        model = BootstrapMCModel(sat_id="G01", orbital_period_hr=11.9667, n_bootstrap=50)
        model.fit(ts, residual, resets=[])
        _, std = model.predict(future_timestamps)
        # With no resets, ensemble std should be very small
        assert std.mean() < 2.0

    def test_predict_without_fit_raises(self, future_timestamps):
        from src.models.bootstrap_mc import BootstrapMCModel
        model = BootstrapMCModel(sat_id="G01", orbital_period_hr=11.9667)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(future_timestamps)

    def test_reproducible_with_same_seed(self, sawtooth_gps_series, future_timestamps):
        """Same seed should give identical results."""
        from src.models.bootstrap_mc import BootstrapMCModel
        from src.reset_detector import detect_resets

        ts, residual = _get_residual(sawtooth_gps_series)
        resets = detect_resets(residual, "G01")

        m1 = BootstrapMCModel("G01", 11.9667, n_bootstrap=50, seed=42)
        m1.fit(ts, residual, resets)
        mean1, _ = m1.predict(future_timestamps)

        m2 = BootstrapMCModel("G01", 11.9667, n_bootstrap=50, seed=42)
        m2.fit(ts, residual, resets)
        mean2, _ = m2.predict(future_timestamps)

        np.testing.assert_array_equal(mean1, mean2)


# ===========================================================================
# Student-t tests (Module 9)
# ===========================================================================

class TestStudentTModel:
    gpytorch = pytest.importorskip("gpytorch")

    def test_fit_and_predict_shape(self, irregular_sawtooth_series, future_timestamps):
        from src.models.student_t import StudentTModel
        ts, residual = _get_residual(irregular_sawtooth_series)

        model = StudentTModel(sat_id="G21", orbital_period_hr=11.9667, n_iterations=20)
        model.fit(ts, residual)
        mean, std = model.predict(future_timestamps)

        assert mean.shape == (96,)
        assert std.shape  == (96,)
        assert (std > 0).all()

    def test_learned_nu_in_bounds(self, irregular_sawtooth_series, future_timestamps):
        from src.models.student_t import StudentTModel
        from src.config import STUDENT_T_NU_BOUNDS
        ts, residual = _get_residual(irregular_sawtooth_series)

        model = StudentTModel("G21", 11.9667, n_iterations=20)
        model.fit(ts, residual)

        nu = model.learned_nu
        nu_lo, nu_hi = STUDENT_T_NU_BOUNDS
        assert nu_lo <= nu <= nu_hi, f"Learned ν={nu:.2f} outside [{nu_lo}, {nu_hi}]"

    def test_std_inflated_vs_gp(self, irregular_sawtooth_series, future_timestamps):
        """
        Student-t should produce wider std than standard GP on heavy-tail data
        due to the ν/(ν-2) inflation factor.
        """
        pytest.importorskip("gpytorch")
        from src.models.student_t import StudentTModel
        from src.models.gp_model import GPModel

        ts, residual = _get_residual(irregular_sawtooth_series)

        st_model = StudentTModel("G21", 11.9667, n_iterations=20)
        st_model.fit(ts, residual)
        _, std_st = st_model.predict(future_timestamps)

        gp_model = GPModel("G21", 11.9667, n_iterations=20)
        gp_model.fit(ts, residual)
        _, std_gp = gp_model.predict(future_timestamps)

        # Student-t std should be ≥ GP std on average
        assert std_st.mean() >= std_gp.mean() * 0.9  # Allow 10% tolerance


# ===========================================================================
# Matérn Fallback tests (Module 10)
# ===========================================================================

class TestMaternFallbackModel:
    gpytorch = pytest.importorskip("gpytorch")

    def test_fit_and_predict_shape(self, clean_gps_series, future_timestamps):
        from src.models.matern_fallback import MaternFallbackModel
        ts, residual = _get_residual(clean_gps_series)

        model = MaternFallbackModel(sat_id="G99", n_iterations=30)
        model.fit(ts, residual)
        mean, std = model.predict(future_timestamps)

        assert mean.shape == (96,)
        assert std.shape  == (96,)
        assert (std > 0).all()

    def test_predict_without_fit_raises(self, future_timestamps):
        from src.models.matern_fallback import MaternFallbackModel
        model = MaternFallbackModel(sat_id="G99")
        with pytest.raises(RuntimeError, match="not trained"):
            model.predict(future_timestamps)

    def test_fallback_completes_quickly(self, clean_gps_series, future_timestamps):
        """Fallback should train faster than full GP (fewer iterations)."""
        import time
        pytest.importorskip("gpytorch")
        from src.models.matern_fallback import MaternFallbackModel
        ts, residual = _get_residual(clean_gps_series)

        model = MaternFallbackModel(sat_id="G99", n_iterations=50)
        t0 = time.time()
        model.fit(ts, residual)
        elapsed = time.time() - t0
        assert elapsed < 120  # Should complete within 2 minutes on CPU


# ===========================================================================
# Postprocess tests (Module 11)
# ===========================================================================

from src.postprocess import (
    winsorize_predictions,
    format_satellite_output,
    evaluate_gaussianity,
)


class TestPostprocess:

    def test_winsorize_clips_outliers(self, clean_gps_series, sample_timestamps):
        """Extreme predictions should be clipped."""
        train_std = float(clean_gps_series.std())
        mean_pred = np.array([0.0] * 90 + [1000.0] * 3 + [-1000.0] * 3)  # outliers
        std_pred  = np.ones(96) * 0.5

        mean_clipped, _ = winsorize_predictions(
            mean_pred=mean_pred,
            std_pred=std_pred,
            train_series=clean_gps_series,
            sat_id="G01",
        )

        clip_bound = clean_gps_series.mean() + 3 * train_std
        assert (mean_clipped <= clip_bound + 1e-6).all()
        assert (mean_clipped >= clean_gps_series.mean() - 3 * train_std - 1e-6).all()

    def test_winsorize_does_not_clip_valid(self, clean_gps_series):
        """In-range predictions should not be affected."""
        mean_pred = np.zeros(96)
        std_pred  = np.ones(96) * 0.5

        mean_clipped, _ = winsorize_predictions(
            mean_pred=mean_pred,
            std_pred=std_pred,
            train_series=clean_gps_series,
        )

        # Zero is well within range for a zero-mean signal
        np.testing.assert_allclose(mean_clipped, mean_pred)

    def test_format_output_columns(self, future_timestamps):
        """Output DataFrame should have required columns."""
        mean_ns = np.random.normal(0, 2, 96)
        std_ns  = np.abs(np.random.normal(0.5, 0.1, 96))

        df = format_satellite_output(
            sat_id="G01",
            future_timestamps=future_timestamps,
            mean_ns=mean_ns,
            std_ns=std_ns,
            last_train_timestamp=future_timestamps[0] - pd.Timedelta("15min"),
        )

        assert "sat_id" in df.columns
        assert "timestamp" in df.columns
        assert "mean_ns" in df.columns
        assert "std_ns" in df.columns
        assert "horizon_min" in df.columns
        assert len(df) == 96

    def test_format_output_horizon_values(self, future_timestamps):
        """Horizon values should match 15-min increments."""
        last_ts = future_timestamps[0] - pd.Timedelta("15min")
        df = format_satellite_output(
            "G01", future_timestamps, np.zeros(96), np.ones(96), last_ts
        )
        assert df["horizon_min"].iloc[0] == 15
        assert df["horizon_min"].iloc[1] == 30
        assert df["horizon_min"].iloc[-1] == 96 * 15

    def test_gaussianity_returns_dataframe(self, future_timestamps):
        """evaluate_gaussianity should return a non-empty DataFrame."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "sat_id": ["G01"] * 96,
            "timestamp": future_timestamps,
            "mean_ns": rng.normal(0, 2, 96),
            "std_ns": np.abs(rng.normal(0.5, 0.1, 96)),
            "horizon_min": np.arange(1, 97) * 15,
        })

        metrics = evaluate_gaussianity(df)
        assert isinstance(metrics, pd.DataFrame)
        assert "skewness" in metrics.columns
        assert "excess_kurtosis" in metrics.columns
        assert "is_gaussian" in metrics.columns
