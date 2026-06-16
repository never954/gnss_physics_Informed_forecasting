"""
generate_synthetic_data.py — Generate synthetic GNSS error CSV for pipeline smoke testing.

Creates a realistic 7-day dataset with:
  - 4 GPS MEO satellites (G01 clean, G02 regular sawtooth, G21 irregular, G31 eclipse)
  - 1 BeiDou GEO satellite (C03 clean)
  - 1 Galileo MEO satellite (E05 regular sawtooth)

Run with:
    python data/raw/sample/generate_synthetic_data.py

Output: data/raw/train/synthetic_gnss_errors.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

def make_series(sat_id, n=672, seed=42, resets=None, eclipse_at=None,
                orbital_period_hr=11.9667, amplitude=2.0, solar_amp=1.0,
                drift=0.01, noise_std=0.3):
    """Generate one satellite's SISE time series."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * 0.25  # hours

    sise = (
        drift * t
        + amplitude * np.sin(2 * np.pi * t / orbital_period_hr + rng.uniform(0, 2*np.pi))
        + solar_amp  * np.cos(2 * np.pi * t / 24.0 + rng.uniform(0, 2*np.pi))
        + rng.normal(0, noise_std, n)
    )

    if resets:
        for reset_idx, mag in resets:
            sise[reset_idx:] += mag

    if eclipse_at:
        for idx in eclipse_at:
            sise[idx] += 25.0
            sise[idx+1] += 10.0 if idx+1 < n else 0

    return sise


def generate(output_path: str = "data/raw/train/synthetic_gnss_errors.csv"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    timestamps = pd.date_range("2024-01-01", periods=672, freq="15min", tz="UTC")
    t_str = timestamps.strftime("%Y-%m-%dT%H:%M:%SZ")

    configs = [
        # (sat_id, seed, resets, eclipses, orbital_period_hr, amplitude, drift)
        ("G01", 42,   None,                    None,        11.9667, 2.0, 0.010),  # clean GPS
        ("G02", 99,   [(100,-5),(300,-5),(500,-5)], None,   11.9667, 2.5, 0.012),  # regular sawtooth
        ("G21", 77,   [(80,-8),(200,12),(450,-6),(580,15)], [350],   11.9667, 3.0, 0.015),  # irregular + eclipse
        ("G31", 55,   [(70,-7),(250,9)],        [400,560],  11.9667, 2.2, 0.011),  # irregular
        ("C03", 33,   None,                    None,        24.0,    3.5, 0.005),  # GEO BeiDou
        ("E05", 88,   [(200,-4),(450,-4)],      None,       14.0833, 1.8, 0.008),  # Galileo regular
    ]

    frames = []
    for sat_id, seed, resets, eclipses, period, amp, drift in configs:
        sise = make_series(
            sat_id=sat_id, n=672, seed=seed,
            resets=resets, eclipse_at=eclipses,
            orbital_period_hr=period, amplitude=amp,
            drift=drift, noise_std=0.3,
        )
        clock_err = sise + np.random.default_rng(seed+1).normal(0, 0.1, 672)
        eph_err   = np.random.default_rng(seed+2).normal(0, 0.3, 672)

        df = pd.DataFrame({
            "timestamp":       t_str,
            "sat_id":          sat_id,
            "clock_error_ns":  clock_err,
            "eph_error_m":     eph_err,
            "sise_ns":         sise,
        })
        frames.append(df)
        print(f"  Generated {sat_id}: {len(df)} points | "
              f"SISE range=[{sise.min():.2f}, {sise.max():.2f}] ns")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output_path, index=False)
    print(f"\nSynthetic data saved to: {output_path}")
    print(f"Total rows: {len(combined)} | Satellites: {combined['sat_id'].nunique()}")
    return combined


if __name__ == "__main__":
    generate()
