"""
test_evaluate.py — Tests for src/evaluate.py.

Covers:
  - Loading and normalising predictions CSV
  - Loading actuals from Module 0 output
  - Merging on (satellite_id, timestamp)
  - Metric computation (MAE, RMSE, MBE, coverage, CRPS)
  - Full integration: evaluate_predictions() with synthetic Day 8 data
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluate import (
    _load_predictions,
    _merge_predictions_actuals,
    _compute_metrics,
    _crps_gaussian,
    evaluate_predictions,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def fake_submission(tmp_path):
    """Minimal submission.csv with 3 satellites × 4 horizons."""
    ts_base = pd.Timestamp("2026-01-08 00:00:00", tz="UTC")
    rows = []
    for sat in ["G01", "E05", "R03"]:
        for i, h in enumerate([15, 30, 60, 120]):
            ts = ts_base + pd.Timedelta(minutes=h - 15)
            rows.append({
                "satellite_id": sat,
                "timestamp": ts.isoformat(),
                "mean_ns": 5.0 + i * 0.5,
                "std_ns":  1.0,
                "horizon_min": h,
            })
    df = pd.DataFrame(rows)
    path = tmp_path / "submission.csv"
    df.to_csv(path, index=False)
    return path, df


@pytest.fixture
def fake_actuals_dir(tmp_path):
    """Per-satellite *_sise.csv files matching the fake_submission timestamps."""
    ts_base = pd.Timestamp("2026-01-08 00:00:00", tz="UTC")
    actuals_dir = tmp_path / "actuals"
    actuals_dir.mkdir()

    for sat in ["G01", "E05", "R03"]:
        rows = []
        for i, h in enumerate([15, 30, 60, 120]):
            ts = ts_base + pd.Timedelta(minutes=h - 15)
            rows.append({
                "satellite_id": sat,
                "timestamp": ts.isoformat(),
                "clock_error_ns": 5.3 + i * 0.5,   # slightly off from predictions
                "eph_error_m": np.nan,
                "sise_ns": 5.3 + i * 0.5,
            })
        pd.DataFrame(rows).to_csv(actuals_dir / f"{sat}_sise.csv", index=False)

    return actuals_dir


@pytest.fixture
def synthetic_day8_csv(tmp_path):
    """
    Create a synthetic 2026_008.csv in the competition 78-column format
    for a single GPS satellite G01 with 2 broadcast epochs.
    """
    gps_week = 2337
    toe_sow  = 388800.0
    rows = []
    for hour in [0, 2]:
        epoch = pd.Timestamp(f"2026-01-08T{hour:02d}:00:00", tz="UTC")
        rows.append({
            "satellite_id": "G01",
            "epoch": epoch.isoformat(),
            "constellation": "GPS",
            "af0": 3.479e-4, "af1": -2.96e-12, "af2": 0.0,
            "toe_sec_gps_week": toe_sow, "gps_week": gps_week,
            "toe_sec_gal_week": np.nan, "gal_week": np.nan,
            "toe_sec_bds_week": np.nan, "bds_week": np.nan,
            "toe_sec_irn_week": np.nan, "irn_week": np.nan,
            "sqrt_a_sqrt_m": 5153.8, "e_eccentricity": 0.01,
            "i0_rad": 0.9613, "omega0_rad": 1.2345, "omega_rad": 2.1,
            "m0_rad": 0.5, "delta_n_rad_sec": 4.6e-9,
            "idot_rad_sec": -1.5e-11, "omega_dot_rad_sec": -7.5e-9,
            "crs_m": 31.125, "crc_m": 265.0,
            "cus_rad": 4.76e-6, "cuc_rad": -2.8e-7,
            "cis_rad": -1.12e-7, "cic_rad": -5.96e-8,
            "igs_clock_bias_seconds": np.nan,
            "igs_x_km": np.nan, "igs_y_km": np.nan, "igs_z_km": np.nan,
        })
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "2026_008.csv"
    df.to_csv(csv_path, index=False)
    return csv_path, tmp_path


# ── Unit tests ────────────────────────────────────────────────────────────────

class TestLoadPredictions:
    def test_loads_correctly(self, fake_submission):
        path, expected = fake_submission
        df = _load_predictions(path)
        assert "satellite_id" in df.columns
        assert "timestamp" in df.columns
        assert "mean_ns" in df.columns
        assert "std_ns" in df.columns
        assert "horizon_min" in df.columns

    def test_sat_id_alias_resolved(self, tmp_path):
        """sat_id column alias should be normalised to satellite_id."""
        rows = [{"sat_id": "G01", "timestamp": "2026-01-08T00:00:00Z",
                 "mean_ns": 1.0, "std_ns": 0.5, "horizon_min": 15}]
        path = tmp_path / "sub.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        df = _load_predictions(path)
        assert "satellite_id" in df.columns
        assert "sat_id" not in df.columns

    def test_timestamp_parsed_as_datetime(self, fake_submission):
        path, _ = fake_submission
        df = _load_predictions(path)
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])


class TestMergePredictionsActuals:
    def test_matched_pairs_found(self, fake_submission, fake_actuals_dir):
        _, preds_df = fake_submission
        preds_df["timestamp"] = pd.to_datetime(preds_df["timestamp"], utc=True)

        actuals_frames = []
        for f in fake_actuals_dir.glob("*_sise.csv"):
            df = pd.read_csv(f)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            actuals_frames.append(df[["satellite_id", "timestamp", "sise_ns"]])
        actuals = pd.concat(actuals_frames, ignore_index=True)

        merged = _merge_predictions_actuals(preds_df, actuals)
        assert len(merged) == 12   # 3 sats × 4 horizons
        assert "error_ns" in merged.columns
        assert "z_score" in merged.columns

    def test_error_computed_correctly(self, fake_submission, fake_actuals_dir):
        """error = actual - predicted."""
        _, preds_df = fake_submission
        preds_df["timestamp"] = pd.to_datetime(preds_df["timestamp"], utc=True)

        actuals_frames = []
        for f in fake_actuals_dir.glob("*_sise.csv"):
            df = pd.read_csv(f)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            actuals_frames.append(df[["satellite_id", "timestamp", "sise_ns"]])
        actuals = pd.concat(actuals_frames, ignore_index=True)

        merged = _merge_predictions_actuals(preds_df, actuals)
        expected_err = (merged["actual_ns"] - merged["mean_ns"]).values
        np.testing.assert_allclose(merged["error_ns"].values, expected_err)

    def test_no_overlap_returns_empty(self):
        """No common timestamps → empty result."""
        preds = pd.DataFrame({
            "satellite_id": ["G01"],
            "timestamp": pd.to_datetime(["2026-01-08T00:00:00Z"], utc=True),
            "mean_ns": [1.0], "std_ns": [0.5], "horizon_min": [15],
        })
        actuals = pd.DataFrame({
            "satellite_id": ["G01"],
            "timestamp": pd.to_datetime(["2026-01-09T00:00:00Z"], utc=True),
            "sise_ns": [2.0],
        })
        merged = _merge_predictions_actuals(preds, actuals)
        assert merged.empty


class TestCRPS:
    def test_perfect_forecast_near_zero(self):
        """Perfect point forecast with tiny std → CRPS ≈ 0."""
        y    = np.array([1.0, 2.0, 3.0])
        mu   = np.array([1.0, 2.0, 3.0])
        sigma = np.array([1e-6, 1e-6, 1e-6])
        crps = _crps_gaussian(y, mu, sigma)
        assert abs(crps) < 1e-4

    def test_larger_error_gives_larger_crps(self):
        """Worse predictions → higher CRPS."""
        y     = np.array([5.0, 5.0, 5.0])
        mu1   = np.array([5.0, 5.0, 5.0])   # perfect
        mu2   = np.array([10.0, 10.0, 10.0]) # wrong by 5 ns
        sigma = np.array([1.0, 1.0, 1.0])
        crps1 = _crps_gaussian(y, mu1, sigma)
        crps2 = _crps_gaussian(y, mu2, sigma)
        assert crps2 > crps1

    def test_overconfident_worse_than_calibrated(self):
        """Overconfident std (too small) → higher CRPS than well-calibrated."""
        y     = np.array([0.0, 2.0, -1.5, 3.0])
        mu    = np.zeros(4)
        sigma_good = np.array([1.5, 1.5, 1.5, 1.5])   # appropriate
        sigma_over = np.array([0.01, 0.01, 0.01, 0.01]) # overconfident
        crps_good = _crps_gaussian(y, mu, sigma_good)
        crps_over = _crps_gaussian(y, mu, sigma_over)
        assert crps_over > crps_good


class TestComputeMetrics:
    def _make_merged(self, actual, predicted, std, sat="G01"):
        """Build a merged DataFrame for testing metrics."""
        n = len(actual)
        ts_base = pd.Timestamp("2026-01-08", tz="UTC")
        horizons = [15, 30, 60, 120, 1440]
        rows = []
        for i in range(n):
            h = horizons[i % len(horizons)]
            ts = ts_base + pd.Timedelta(minutes=h - 15)
            rows.append({
                "satellite_id": sat,
                "timestamp": ts,
                "timestamp_key": ts,
                "actual_ns":  actual[i],
                "mean_ns":    predicted[i],
                "std_ns":     std[i],
                "horizon_min": h,
                "error_ns":  actual[i] - predicted[i],
                "z_score":   (actual[i] - predicted[i]) / std[i],
                "abs_error": abs(actual[i] - predicted[i]),
                "sq_error":  (actual[i] - predicted[i]) ** 2,
                "within_1sigma": int(abs((actual[i] - predicted[i]) / std[i]) <= 1),
                "within_2sigma": int(abs((actual[i] - predicted[i]) / std[i]) <= 2),
            })
        return pd.DataFrame(rows)

    def test_zero_error_gives_zero_mae_rmse(self):
        """Perfect predictions → MAE = RMSE = 0."""
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        merged = self._make_merged(actual, actual, np.ones(5))
        metrics = _compute_metrics(merged)
        all_row = metrics[(metrics["satellite_id"] == "ALL") & (metrics["horizon"] == "all")]
        assert all_row["mae_ns"].values[0] == pytest.approx(0.0, abs=1e-9)
        assert all_row["rmse_ns"].values[0] == pytest.approx(0.0, abs=1e-9)

    def test_coverage_100_when_all_within_1sigma(self):
        """All actuals within ±1σ → cov_1sigma = 100%."""
        # predicted=0, std=100, actual=1 → z=0.01 < 1
        actual = np.array([1.0, -1.0, 0.5, -0.5, 0.0])
        pred   = np.zeros(5)
        std    = np.full(5, 100.0)
        merged = self._make_merged(actual, pred, std)
        metrics = _compute_metrics(merged)
        all_row = metrics[(metrics["satellite_id"] == "ALL") & (metrics["horizon"] == "all")]
        assert all_row["cov_1sigma_%"].values[0] == pytest.approx(100.0)

    def test_mbe_sign_correct(self):
        """Systematic over-prediction → negative MBE (actual < predicted)."""
        actual = np.ones(5)
        pred   = np.full(5, 10.0)   # over-predict by 9 ns
        std    = np.ones(5)
        merged = self._make_merged(actual, pred, std)
        metrics = _compute_metrics(merged)
        all_row = metrics[(metrics["satellite_id"] == "ALL") & (metrics["horizon"] == "all")]
        mbe = all_row["mbe_ns"].values[0]
        assert mbe < 0, f"Over-prediction should give negative MBE, got {mbe}"

    def test_output_has_required_columns(self):
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        merged = self._make_merged(actual, actual, np.ones(5))
        metrics = _compute_metrics(merged)
        required = {"satellite_id", "horizon", "n", "mae_ns", "rmse_ns",
                    "mbe_ns", "cov_1sigma_%", "cov_2sigma_%", "crps"}
        assert required.issubset(set(metrics.columns))

    def test_all_satellites_row_present(self):
        """Metrics must include an 'ALL' aggregate satellite row."""
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        merged = self._make_merged(actual, actual + 0.1, np.ones(5))
        metrics = _compute_metrics(merged)
        assert "ALL" in metrics["satellite_id"].values


class TestEvaluatePredictions:
    def test_integration_runs_and_returns_dataframes(
        self, fake_submission, fake_actuals_dir
    ):
        """evaluate_predictions should return two non-empty DataFrames."""
        preds_path, _ = fake_submission

        # Patch _load_actuals to use fake_actuals_dir directly
        # (bypasses Module 0 for this integration test)
        from unittest.mock import patch
        import src.evaluate as ev_module

        def fake_load_actuals(path):
            frames = []
            for f in fake_actuals_dir.glob("*_sise.csv"):
                df = pd.read_csv(f)
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                frames.append(df[["satellite_id", "timestamp", "sise_ns"]])
            return pd.concat(frames, ignore_index=True)

        with patch.object(ev_module, "_load_actuals", fake_load_actuals), \
             patch.object(ev_module, "prepare_pipeline_input", return_value=fake_actuals_dir):
            with tempfile.TemporaryDirectory() as out_dir:
                summary, detail = ev_module.evaluate_predictions(
                    predictions_csv=preds_path,
                    actual_data_path=fake_actuals_dir,
                    output_csv=Path(out_dir) / "report.csv",
                )
        assert not summary.empty, "Summary DataFrame should not be empty"
        assert not detail.empty,  "Detail DataFrame should not be empty"

    def test_report_csv_created(self, fake_submission, fake_actuals_dir):
        """Output CSV must be written to disk."""
        preds_path, _ = fake_submission
        import src.evaluate as ev_module
        from unittest.mock import patch

        def fake_load_actuals(path):
            frames = []
            for f in fake_actuals_dir.glob("*_sise.csv"):
                df = pd.read_csv(f)
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                frames.append(df[["satellite_id", "timestamp", "sise_ns"]])
            return pd.concat(frames, ignore_index=True)

        with patch.object(ev_module, "_load_actuals", fake_load_actuals), \
             patch.object(ev_module, "prepare_pipeline_input", return_value=fake_actuals_dir):
            with tempfile.TemporaryDirectory() as out_dir:
                report_path = Path(out_dir) / "report.csv"
                ev_module.evaluate_predictions(
                    predictions_csv=preds_path,
                    actual_data_path=fake_actuals_dir,
                    output_csv=report_path,
                )
                assert report_path.exists(), "Report CSV must be created"
                saved = pd.read_csv(report_path)
                assert len(saved) > 0, "Report CSV must not be empty"
