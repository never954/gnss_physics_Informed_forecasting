"""
test_data_loader.py — Tests for Module 2: data_loader.py
"""

import io
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import os

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import (
    load_satellite_data,
    get_train_test_split,
    describe_dataset,
    _resolve_column,
    _interpolate_small_gaps,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_csv_file(tmpdir: str, sat_id: str, n_points: int = 672,
                   clock_col: str = "clock_error_ns",
                   ts_col: str = "timestamp") -> str:
    """Create a synthetic satellite CSV file."""
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2024-01-01", periods=n_points, freq="15min", tz="UTC")
    df = pd.DataFrame({
        ts_col: timestamps.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sat_id": sat_id,
        clock_col: rng.normal(0, 2.0, n_points),
        "eph_error_m": rng.normal(0, 0.5, n_points),
    })
    path = os.path.join(tmpdir, f"{sat_id}.csv")
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadSatelliteData:

    def test_load_single_file(self, tmp_path):
        """Load a single CSV with standard column names."""
        _make_csv_file(str(tmp_path), "G01")
        result = load_satellite_data(tmp_path)

        assert "G01" in result
        df = result["G01"]
        assert len(df) == 672
        assert "sise_ns" in df.columns
        assert "clock_error_ns" in df.columns

    def test_load_multiple_satellites_single_file(self, tmp_path):
        """Single CSV with multiple satellites."""
        rng = np.random.default_rng(0)
        timestamps = pd.date_range("2024-01-01", periods=100, freq="15min", tz="UTC")
        frames = []
        for sat_id in ["G01", "G02", "E01"]:
            frames.append(pd.DataFrame({
                "timestamp": timestamps.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "sat_id": sat_id,
                "clock_error_ns": rng.normal(0, 2.0, 100),
                "eph_error_m": rng.normal(0, 0.5, 100),
            }))
        combined = pd.concat(frames)
        combined.to_csv(tmp_path / "all_sats.csv", index=False)

        result = load_satellite_data(tmp_path)
        assert set(result.keys()) == {"G01", "G02", "E01"}

    def test_column_alias_resolution(self, tmp_path):
        """Test that column aliases are resolved correctly."""
        rng = np.random.default_rng(1)
        n = 100
        timestamps = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
        df = pd.DataFrame({
            "epoch": timestamps.strftime("%Y-%m-%dT%H:%M:%SZ"),  # alias for 'timestamp'
            "prn": "G03",                                          # alias for 'sat_id'
            "dclk_ns": rng.normal(0, 2.0, n),                     # alias for 'clock_error_ns'
        })
        df.to_csv(tmp_path / "aliased.csv", index=False)

        result = load_satellite_data(tmp_path)
        assert "G03" in result

    def test_sise_column_used_directly(self, tmp_path):
        """If sise_ns column exists, use it directly without computing."""
        rng = np.random.default_rng(2)
        n = 200
        timestamps = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
        df = pd.DataFrame({
            "timestamp": timestamps.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sat_id": "C21",
            "sise_ns": rng.normal(0, 3.0, n),
        })
        df.to_csv(tmp_path / "with_sise.csv", index=False)

        result = load_satellite_data(tmp_path)
        assert "C21" in result
        assert "sise_ns" in result["C21"].columns

    def test_no_csv_raises(self, tmp_path):
        """Empty directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_satellite_data(tmp_path)

    def test_missing_error_column_raises(self, tmp_path):
        """CSV with no usable error column skips satellite (logs error)."""
        df = pd.DataFrame({
            "timestamp": ["2024-01-01T00:00:00Z"],
            "sat_id": ["G99"],
            "some_other_column": [1.0],
        })
        df.to_csv(tmp_path / "bad.csv", index=False)
        result = load_satellite_data(tmp_path)
        # G99 should be skipped
        assert "G99" not in result


class TestInterpolation:

    def test_small_gap_filled(self):
        """Small NaN gaps (≤2 steps) should be interpolated."""
        ts = pd.date_range("2024-01-01", periods=10, freq="15min", tz="UTC")
        vals = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], index=ts)
        df = pd.DataFrame({"sise_ns": vals, "clock_error_ns": vals, "eph_error_m": vals * 0.1})

        result = _interpolate_small_gaps(df, max_gap=2)
        assert result["sise_ns"].isna().sum() == 0

    def test_large_gap_not_filled(self):
        """Large NaN gaps (>2 steps) should remain NaN."""
        ts = pd.date_range("2024-01-01", periods=10, freq="15min", tz="UTC")
        vals = pd.Series([1.0, 2.0, np.nan, np.nan, np.nan, np.nan, 7.0, 8.0, 9.0, 10.0], index=ts)
        df = pd.DataFrame({"sise_ns": vals, "clock_error_ns": vals, "eph_error_m": vals * 0.1})

        result = _interpolate_small_gaps(df, max_gap=2)
        assert result["sise_ns"].isna().sum() == 4  # Gap of 4 remains


class TestColumnResolution:

    def test_exact_match(self):
        df = pd.DataFrame({"timestamp": [], "sat_id": []})
        assert _resolve_column(df, ["timestamp"], "timestamp") == "timestamp"

    def test_alias_match(self):
        df = pd.DataFrame({"epoch": [], "prn": []})
        assert _resolve_column(df, ["timestamp", "epoch"], "timestamp") == "epoch"

    def test_case_insensitive(self):
        df = pd.DataFrame({"TIMESTAMP": []})
        assert _resolve_column(df, ["timestamp"], "timestamp") == "TIMESTAMP"

    def test_no_match_raises(self):
        df = pd.DataFrame({"foo": [], "bar": []})
        with pytest.raises(ValueError, match="Could not find column"):
            _resolve_column(df, ["timestamp", "epoch"], "timestamp")


class TestGetTrainTestSplit:

    def test_future_timestamps_count(self, tmp_path):
        """Each satellite should get exactly 96 future timestamps."""
        _make_csv_file(str(tmp_path), "G01")
        sat_data = load_satellite_data(tmp_path)
        _, future_ts = get_train_test_split(sat_data)
        assert len(future_ts["G01"]) == 96

    def test_future_timestamps_cadence(self, tmp_path):
        """Future timestamps should be 15-min apart."""
        _make_csv_file(str(tmp_path), "G01")
        sat_data = load_satellite_data(tmp_path)
        _, future_ts = get_train_test_split(sat_data)
        ts = future_ts["G01"]
        diffs = pd.Series(ts).diff().dropna()
        assert (diffs == pd.Timedelta("15min")).all()

    def test_future_starts_after_training(self, tmp_path):
        """First future timestamp should be 15min after last training point."""
        _make_csv_file(str(tmp_path), "G01")
        sat_data = load_satellite_data(tmp_path)
        train_data, future_ts = get_train_test_split(sat_data)
        last_train = train_data["G01"].index[-1]
        first_future = future_ts["G01"][0]
        assert first_future == last_train + pd.Timedelta("15min")


class TestDescribeDataset:

    def test_returns_dataframe(self, tmp_path):
        _make_csv_file(str(tmp_path), "G01")
        sat_data = load_satellite_data(tmp_path)
        desc = describe_dataset(sat_data)
        assert isinstance(desc, pd.DataFrame)
        assert "G01" in desc.index
        assert "n_points" in desc.columns
        assert "mean_ns" in desc.columns
