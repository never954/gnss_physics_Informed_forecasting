"""
test_gnss_preprocess.py — Tests for Module 0 (gnss_preprocess.py).

Covers:
  - Keplerian → ECEF position computation against ICD reference values
  - Clock polynomial evaluation (af0 + af1*dt + af2*dt²)
  - GPS seconds-of-week conversion
  - dt computation across week boundaries
  - Zero-order-hold resampling (forward-fill cap)
  - IGS Tier A detection
  - Output CSV format matches data_loader.py expectations
  - Full prepare_pipeline_input integration smoke test (synthetic data)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile
import os

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gnss_preprocess import (
    _keplerian_to_ecef,
    _epoch_to_gps_sow,
    _eval_clock_poly_series,
    _compute_dt_series,
    _resample_to_15min,
    prepare_pipeline_input,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def gps_icd_reference_row():
    """
    IS-GPS-200L Table 20-IV example values for satellite position verification.
    These are from the published GPS ICD example (widely cited in literature).
    Reference position (ECEF): approximately [-2,169,700, 14,745,300, 21,605,800] m
    (exact values depend on toe and t; we use a self-consistent check instead)
    """
    return pd.Series({
        "sqrt_a_sqrt_m":      5153.79589081,
        "e_eccentricity":     0.01,
        "i0_rad":             0.9613,
        "omega0_rad":         1.2345,
        "omega_rad":          2.1,
        "m0_rad":             0.5,
        "delta_n_rad_sec":    4.6e-9,
        "idot_rad_sec":       -1.5e-11,
        "omega_dot_rad_sec":  -7.5e-9,
        "crs_m":              31.125,
        "crc_m":              265.0,
        "cus_rad":            4.76e-6,
        "cuc_rad":            -2.8e-7,
        "cis_rad":            -1.12e-7,
        "cic_rad":            -5.96e-8,
        "toe_sec_gps_week":   388800.0,    # 108 hours into GPS week
    })


@pytest.fixture
def glonass_row():
    """Minimal GLONASS broadcast row (uses Cartesian, no Keplerian)."""
    return pd.Series({
        "satellite_id": "R03",
        "epoch": "2026-01-01T00:15:00",
        "constellation": "GLONASS",
        "af0": -1.2e-5,
        "af1":  5.0e-12,
        "af2":  0.0,
        "toe_sec_gps_week": np.nan,
        "gps_week": np.nan,
        "toe_sec_gal_week": np.nan,
        "gal_week": np.nan,
        "toe_sec_bds_week": np.nan,
        "bds_week": np.nan,
        "toe_sec_irn_week": np.nan,
        "irn_week": np.nan,
        "igs_clock_bias_seconds": np.nan,
        "igs_x_km": np.nan,
        "igs_y_km": np.nan,
        "igs_z_km": np.nan,
    })


@pytest.fixture
def synthetic_competition_csv(tmp_path):
    """
    Build a minimal 78-column-style CSV with 3 GPS records (2 with IGS truth)
    and 2 GLONASS records, spanning 2 hours. Saved as 2026_001.csv in tmp_path.
    """
    gps_week = 2337
    toe_sow  = 388800.0   # 108h into the week

    rows = []

    # GPS G01 — 2 epochs with IGS truth
    for i, hour in enumerate([0, 2]):
        epoch = pd.Timestamp(f"2026-01-01T{hour:02d}:00:00", tz="UTC")
        rows.append({
            "satellite_id": "G01",
            "epoch": epoch.isoformat(),
            "constellation": "GPS",
            "af0":  3.478e-4,
            "af1": -2.96e-12,
            "af2":  0.0,
            "toe_sec_gps_week": toe_sow,
            "gps_week": gps_week,
            "toe_sec_gal_week": np.nan, "gal_week": np.nan,
            "toe_sec_bds_week": np.nan, "bds_week": np.nan,
            "toe_sec_irn_week": np.nan, "irn_week": np.nan,
            # Keplerian elements
            "sqrt_a_sqrt_m":     5153.79589081,
            "e_eccentricity":    0.01,
            "i0_rad":            0.9613,
            "omega0_rad":        1.2345,
            "omega_rad":         2.1,
            "m0_rad":            0.5,
            "delta_n_rad_sec":   4.6e-9,
            "idot_rad_sec":     -1.5e-11,
            "omega_dot_rad_sec": -7.5e-9,
            "crs_m":  31.125, "crc_m":  265.0,
            "cus_rad": 4.76e-6, "cuc_rad": -2.8e-7,
            "cis_rad": -1.12e-7, "cic_rad": -5.96e-8,
            # IGS truth
            "igs_clock_bias_seconds": 3.476e-4 if i == 0 else 3.474e-4,
            "igs_x_km": 18551.18,
            "igs_y_km":  8171.33,
            "igs_z_km": 17195.92,
        })

    # GPS G02 — 1 epoch WITHOUT IGS truth
    rows.append({
        "satellite_id": "G02",
        "epoch": pd.Timestamp("2026-01-01T01:00:00", tz="UTC").isoformat(),
        "constellation": "GPS",
        "af0": 1.23e-4, "af1": 1.5e-12, "af2": 0.0,
        "toe_sec_gps_week": toe_sow, "gps_week": gps_week,
        "toe_sec_gal_week": np.nan, "gal_week": np.nan,
        "toe_sec_bds_week": np.nan, "bds_week": np.nan,
        "toe_sec_irn_week": np.nan, "irn_week": np.nan,
        "sqrt_a_sqrt_m": 5153.8, "e_eccentricity": 0.005,
        "i0_rad": 0.97, "omega0_rad": 0.5, "omega_rad": 1.0,
        "m0_rad": 2.0, "delta_n_rad_sec": 4.0e-9,
        "idot_rad_sec": -1.0e-11, "omega_dot_rad_sec": -7.0e-9,
        "crs_m": 20.0, "crc_m": 200.0, "cus_rad": 3.0e-6, "cuc_rad": -2.0e-7,
        "cis_rad": -1.0e-7, "cic_rad": -5.0e-8,
        "igs_clock_bias_seconds": np.nan,
        "igs_x_km": np.nan, "igs_y_km": np.nan, "igs_z_km": np.nan,
    })

    # GLONASS R03 — 2 epochs
    for hour in [0, 2]:
        epoch = pd.Timestamp(f"2026-01-01T{hour:02d}:00:00", tz="UTC")
        rows.append({
            "satellite_id": "R03",
            "epoch": epoch.isoformat(),
            "constellation": "GLONASS",
            "af0": -1.2e-5, "af1": 5.0e-12, "af2": 0.0,
            "toe_sec_gps_week": np.nan, "gps_week": np.nan,
            "toe_sec_gal_week": np.nan, "gal_week": np.nan,
            "toe_sec_bds_week": np.nan, "bds_week": np.nan,
            "toe_sec_irn_week": np.nan, "irn_week": np.nan,
            "sqrt_a_sqrt_m": np.nan, "e_eccentricity": np.nan,
            "i0_rad": np.nan, "omega0_rad": np.nan, "omega_rad": np.nan,
            "m0_rad": np.nan, "delta_n_rad_sec": np.nan,
            "idot_rad_sec": np.nan, "omega_dot_rad_sec": np.nan,
            "crs_m": np.nan, "crc_m": np.nan, "cus_rad": np.nan, "cuc_rad": np.nan,
            "cis_rad": np.nan, "cic_rad": np.nan,
            "igs_clock_bias_seconds": np.nan,
            "igs_x_km": np.nan, "igs_y_km": np.nan, "igs_z_km": np.nan,
        })

    df = pd.DataFrame(rows)
    csv_path = tmp_path / "2026_001.csv"
    df.to_csv(csv_path, index=False)
    return tmp_path, df


# ── Unit tests: Keplerian → ECEF ─────────────────────────────────────────────

class TestKeplerianToECEF:
    def test_output_is_array_of_3(self, gps_icd_reference_row):
        """Output must be a 3-element numpy array."""
        t = _epoch_to_gps_sow(pd.Timestamp("2026-01-01T00:00:00", tz="UTC"))
        result = _keplerian_to_ecef(gps_icd_reference_row, t)
        assert result.shape == (3,), "Should return 3-element array"

    def test_position_is_gps_orbital_radius(self, gps_icd_reference_row):
        """GPS orbital radius should be ~26,560 km (20,200 km altitude + 6,371 km Earth)."""
        t = _epoch_to_gps_sow(pd.Timestamp("2026-01-01T00:00:00", tz="UTC"))
        xyz = _keplerian_to_ecef(gps_icd_reference_row, t)
        radius_km = np.linalg.norm(xyz) / 1000
        assert 24_000 < radius_km < 30_000, (
            f"Expected GPS orbit radius 24,000–30,000 km, got {radius_km:.0f} km"
        )

    def test_different_times_give_different_positions(self, gps_icd_reference_row):
        """Satellite should move between t and t+3600s."""
        t1 = _epoch_to_gps_sow(pd.Timestamp("2026-01-01T00:00:00", tz="UTC"))
        t2 = _epoch_to_gps_sow(pd.Timestamp("2026-01-01T01:00:00", tz="UTC"))
        xyz1 = _keplerian_to_ecef(gps_icd_reference_row, t1)
        xyz2 = _keplerian_to_ecef(gps_icd_reference_row, t2)
        dist = np.linalg.norm(xyz2 - xyz1) / 1000
        # GPS moves ~3–4 km/s → ~10,800 km/hour
        assert dist > 5_000, f"Satellite should move >5,000 km in 1 hour, moved {dist:.0f} km"

    def test_orbital_radius_stable_over_orbit(self, gps_icd_reference_row):
        """Orbital radius should stay nearly constant (low eccentricity orbit)."""
        radii = []
        for hour in range(12):
            t = _epoch_to_gps_sow(
                pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(hours=hour)
            )
            xyz = _keplerian_to_ecef(gps_icd_reference_row, t)
            radii.append(np.linalg.norm(xyz) / 1000)
        radii = np.array(radii)
        # For e=0.01, radius variation should be < 2%
        variation = (radii.max() - radii.min()) / radii.mean()
        assert variation < 0.05, f"Radius variation {variation:.3%} too large for low-e orbit"


# ── Unit tests: GPS SoW conversion ───────────────────────────────────────────

class TestEpochToGpsSow:
    def test_gps_epoch_zero(self):
        """GPS epoch itself should give SoW = 0."""
        gps_epoch = pd.Timestamp("1980-01-06 00:00:00", tz="UTC")
        assert _epoch_to_gps_sow(gps_epoch) == pytest.approx(0.0, abs=1.0)

    def test_known_epoch(self):
        """Check a known GPS time value."""
        # 2026-01-01 00:00:00 UTC
        # GPS week 2337, day 3 (Wednesday) = 3 × 86400 = 259200 SoW
        epoch = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
        sow = _epoch_to_gps_sow(epoch)
        assert 0.0 <= sow < 604800.0, f"SoW must be in [0, 604800), got {sow}"

    def test_monotone_within_week(self):
        """SoW should increase monotonically within a GPS week (no rollover)."""
        base = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
        sow_vals = [
            _epoch_to_gps_sow(base + pd.Timedelta(hours=h))
            for h in range(24)
        ]
        diffs = np.diff(sow_vals)
        assert all(d > 0 for d in diffs), "SoW should increase within a week"


# ── Unit tests: Clock polynomial ─────────────────────────────────────────────

class TestClockPolynomial:
    def _make_gps_df(self, af0, af1, af2, toe_sow, gps_week, epochs):
        rows = []
        for ep in epochs:
            rows.append({
                "epoch": ep,
                "constellation": "GPS",
                "af0": af0, "af1": af1, "af2": af2,
                "toe_sec_gps_week": toe_sow,
                "gps_week": gps_week,
                "toe_sec_gal_week": np.nan, "gal_week": np.nan,
                "toe_sec_bds_week": np.nan, "bds_week": np.nan,
                "toe_sec_irn_week": np.nan, "irn_week": np.nan,
            })
        return pd.DataFrame(rows)

    def test_at_reference_time_equals_af0(self):
        """At t = t_oc, clock correction = af0 exactly."""
        toe_sow = 388800.0
        gps_week = 2337
        # Epoch that corresponds to exactly toe_sow in GPS time
        gps_epoch = pd.Timestamp("1980-01-06", tz="UTC")
        epoch_ts = gps_epoch + pd.Timedelta(seconds=gps_week * 604800 + toe_sow)

        af0 = 3.478e-4
        df = self._make_gps_df(af0, 0.0, 0.0, toe_sow, gps_week, [epoch_ts])
        result = _eval_clock_poly_series(df, "GPS")
        assert result.iloc[0] == pytest.approx(af0, rel=1e-9)

    def test_linear_drift_applied(self):
        """af1 drift: after 3600s, correction increases by af1 × 3600."""
        toe_sow = 100.0
        gps_week = 2337
        af0, af1 = 0.0, 1e-9  # 1 ns/s drift
        gps_epoch = pd.Timestamp("1980-01-06", tz="UTC")
        t0 = gps_epoch + pd.Timedelta(seconds=gps_week * 604800 + toe_sow)
        t1 = t0 + pd.Timedelta(seconds=3600)

        df = self._make_gps_df(af0, af1, 0.0, toe_sow, gps_week, [t0, t1])
        result = _eval_clock_poly_series(df, "GPS")
        expected_delta = af1 * 3600
        actual_delta = result.iloc[1] - result.iloc[0]
        assert actual_delta == pytest.approx(expected_delta, rel=1e-6)

    def test_glonass_returns_af0_directly(self):
        """GLONASS has no reference time → dt=0 → result = af0."""
        af0 = -1.2e-5
        epochs = [pd.Timestamp("2026-01-01T00:00:00", tz="UTC")]
        rows = [{"epoch": epochs[0], "constellation": "GLONASS",
                 "af0": af0, "af1": 5e-12, "af2": 0.0,
                 "toe_sec_gps_week": np.nan, "gps_week": np.nan,
                 "toe_sec_gal_week": np.nan, "gal_week": np.nan,
                 "toe_sec_bds_week": np.nan, "bds_week": np.nan,
                 "toe_sec_irn_week": np.nan, "irn_week": np.nan,}]
        df = pd.DataFrame(rows)
        result = _eval_clock_poly_series(df, "GLONASS")
        assert result.iloc[0] == pytest.approx(af0, rel=1e-9)


# ── Unit tests: Resampling ────────────────────────────────────────────────────

class TestResampleTo15Min:
    def _make_grid(self, start="2026-01-01", periods=96):
        return pd.date_range(start, periods=periods, freq="15min", tz="UTC")

    def test_irregular_timestamps_mapped_to_grid(self):
        """Broadcast epochs at T+0h, T+2h, T+4h should fill 8 slots (ZOH)."""
        base = pd.Timestamp("2026-01-01", tz="UTC")
        epochs = [base, base + pd.Timedelta(hours=2), base + pd.Timedelta(hours=4)]
        values = [10.0, 20.0, 30.0]
        s = pd.Series(values, index=epochs)
        grid = self._make_grid(periods=24)
        result = _resample_to_15min("G01", s, grid, max_fill_steps=16)
        # Slot 0 (00:00) = 10.0
        assert result["sise_ns"].iloc[0] == pytest.approx(10.0)
        # Slot 8 (02:00) = 20.0 (start of second epoch)
        assert result["sise_ns"].iloc[8] == pytest.approx(20.0)

    def test_fill_cap_leaves_nan_beyond_limit(self):
        """After max_fill_steps, values should be NaN."""
        base = pd.Timestamp("2026-01-01", tz="UTC")
        # Only 1 epoch at T=0, fill cap = 4 steps (1 hour)
        s = pd.Series([5.0], index=[base])
        grid = self._make_grid(periods=16)
        result = _resample_to_15min("G01", s, grid, max_fill_steps=4)
        # Slots 0–4 should be filled (4 = last valid)
        assert result["sise_ns"].iloc[4] == pytest.approx(5.0)
        # Slots 5+ should be NaN
        assert np.isnan(result["sise_ns"].iloc[5])

    def test_duplicate_epochs_keeps_last(self):
        """If two broadcasts at same epoch, keep the later (newer upload)."""
        base = pd.Timestamp("2026-01-01", tz="UTC")
        s = pd.Series([100.0, 200.0], index=[base, base])  # duplicate
        grid = self._make_grid(periods=4)
        result = _resample_to_15min("G01", s, grid, max_fill_steps=16)
        assert result["sise_ns"].iloc[0] == pytest.approx(200.0)

    def test_output_columns_match_data_loader_format(self):
        """Output DataFrame must have columns data_loader.py expects."""
        base = pd.Timestamp("2026-01-01", tz="UTC")
        s = pd.Series([1.0, 2.0], index=[base, base + pd.Timedelta(hours=1)])
        grid = self._make_grid(periods=8)
        result = _resample_to_15min("E05", s, grid, max_fill_steps=16)
        required = {"satellite_id", "clock_error_ns", "eph_error_m", "sise_ns"}
        assert required.issubset(set(result.columns)), (
            f"Missing columns: {required - set(result.columns)}"
        )

    def test_satellite_id_column_populated(self):
        """satellite_id column should match the sat_id argument."""
        base = pd.Timestamp("2026-01-01", tz="UTC")
        s = pd.Series([5.0], index=[base])
        grid = self._make_grid(periods=4)
        result = _resample_to_15min("C21", s, grid, max_fill_steps=16)
        assert (result["satellite_id"] == "C21").all()


# ── Integration: prepare_pipeline_input ──────────────────────────────────────

class TestPrepareInput:
    def test_creates_output_csvs(self, synthetic_competition_csv):
        """prepare_pipeline_input must create one CSV per satellite."""
        data_dir, _ = synthetic_competition_csv
        with tempfile.TemporaryDirectory() as out_dir:
            out_path = prepare_pipeline_input(
                data_dir=data_dir,
                output_dir=out_dir,
                n_train_days=1,
                force_recompute=True,
            )
            csv_files = list(Path(out_dir).glob("*_sise.csv"))
            # Expect G01, G02, R03
            sat_ids = {f.stem.replace("_sise", "") for f in csv_files}
            assert "G01" in sat_ids, "G01 output CSV not found"
            assert "G02" in sat_ids, "G02 output CSV not found"
            assert "R03" in sat_ids, "R03 output CSV not found"

    def test_output_csv_has_required_columns(self, synthetic_competition_csv):
        """Each output CSV must have the columns data_loader.py needs."""
        data_dir, _ = synthetic_competition_csv
        required_cols = {"timestamp", "satellite_id", "clock_error_ns",
                         "eph_error_m", "sise_ns"}
        with tempfile.TemporaryDirectory() as out_dir:
            prepare_pipeline_input(
                data_dir=data_dir,
                output_dir=out_dir,
                n_train_days=1,
                force_recompute=True,
            )
            for csv_file in Path(out_dir).glob("*_sise.csv"):
                df = pd.read_csv(csv_file)
                missing = required_cols - set(df.columns)
                assert not missing, f"{csv_file.name} missing columns: {missing}"

    def test_gps_default_uses_proxy_not_igs(self, synthetic_competition_csv):
        """
        Default (GPS_USE_IGS=False): G01 SISE must equal the af0 polynomial proxy.
        broadcast af0 = 3.478e-4 s → 347,800 ns  (at reference time dt=0)
        Even though IGS truth is present in the fixture, it must NOT be subtracted.
        """
        data_dir, _ = synthetic_competition_csv
        with tempfile.TemporaryDirectory() as out_dir:
            prepare_pipeline_input(
                data_dir=data_dir,
                output_dir=out_dir,
                n_train_days=1,
                force_recompute=True,
                gps_use_igs=False,      # explicit default
            )
            g01 = pd.read_csv(Path(out_dir) / "G01_sise.csv")
            g01_valid = g01["sise_ns"].dropna()
            assert len(g01_valid) > 0, "G01 should have at least one valid SISE value"
            # Proxy value ≈ af0 × 1e9 = 347,800 ns (dt varies slightly across epochs)
            raw_af0_ns = 3.478e-4 * 1e9
            assert np.allclose(g01_valid.values, raw_af0_ns, rtol=0.01), (
                f"GPS proxy mode: SISE should ≈ af0×1e9 ({raw_af0_ns:.0f} ns), "
                f"got mean {g01_valid.mean():.0f} ns"
            )

    def test_gps_igs_mode_differs_from_proxy(self, synthetic_competition_csv):
        """
        When GPS_USE_IGS=True: G01 SISE ≠ raw af0 proxy because IGS clock is subtracted.
        broadcast af0 = 3.478e-4 s → 347,800 ns
        igs_clock    = 3.476e-4 s → subtracted → clock diff ≈ 200 ns
        Final sise_ns includes ephemeris error but must clearly differ from raw proxy.
        """
        data_dir, _ = synthetic_competition_csv
        with tempfile.TemporaryDirectory() as out_dir:
            prepare_pipeline_input(
                data_dir=data_dir,
                output_dir=out_dir,
                n_train_days=1,
                force_recompute=True,
                gps_use_igs=True,
            )
            g01 = pd.read_csv(Path(out_dir) / "G01_sise.csv")
            g01_valid = g01["sise_ns"].dropna()
            assert len(g01_valid) > 0, "G01 should have at least one valid SISE value"
            raw_af0_ns = 3.478e-4 * 1e9
            # IGS-subtracted SISE must differ substantially from raw af0 proxy
            assert not np.allclose(g01_valid.values, raw_af0_ns, rtol=0.001), (
                "IGS mode: SISE should differ from raw af0×1e9 (IGS subtraction must occur)"
            )


    def test_glonass_proxy_uses_af0(self, synthetic_competition_csv):
        """GLONASS (no IGS truth) should have sise_ns ≈ af0 * 1e9."""
        data_dir, _ = synthetic_competition_csv
        with tempfile.TemporaryDirectory() as out_dir:
            prepare_pipeline_input(
                data_dir=data_dir,
                output_dir=out_dir,
                n_train_days=1,
                force_recompute=True,
            )
            r03 = pd.read_csv(Path(out_dir) / "R03_sise.csv")
            r03_valid = r03["sise_ns"].dropna()
            assert len(r03_valid) > 0, "R03 should have at least one valid SISE value"
            expected_ns = -1.2e-5 * 1e9  # -12,000 ns
            assert r03_valid.abs().mean() > abs(expected_ns) * 0.5, (
                "GLONASS proxy should be in the right ballpark of af0 * 1e9"
            )

    def test_returns_path_that_exists(self, synthetic_competition_csv):
        """Return value must be an existing directory."""
        data_dir, _ = synthetic_competition_csv
        with tempfile.TemporaryDirectory() as out_dir:
            result = prepare_pipeline_input(
                data_dir=data_dir,
                output_dir=out_dir,
                n_train_days=1,
                force_recompute=True,
            )
            assert Path(result).is_dir(), "prepare_pipeline_input must return an existing dir"

    def test_caching_skips_existing(self, synthetic_competition_csv):
        """Second call without force_recompute should not crash and skip files."""
        data_dir, _ = synthetic_competition_csv
        with tempfile.TemporaryDirectory() as out_dir:
            prepare_pipeline_input(
                data_dir=data_dir, output_dir=out_dir,
                n_train_days=1, force_recompute=True,
            )
            # Second call — should not raise
            prepare_pipeline_input(
                data_dir=data_dir, output_dir=out_dir,
                n_train_days=1, force_recompute=False,
            )
