"""Robust ingestion for the PS-08 error files.

Assumes almost nothing about sampling: the only hard requirement is a parseable
timestamp column plus the four error channels. Everything else (interval, period)
is derived downstream. Returns continuous time so irregular sampling is native.
"""
import numpy as np
import pandas as pd

from config import CHANNELS, TIME_COL, TIME_FMT


def _norm_cols(df):
    df.columns = [" ".join(str(c).split()) for c in df.columns]
    return df


def load_series(path):
    """Load one file -> (times: datetime64[ns] array, values: {channel: float array}).

    Steps: normalise column whitespace, parse time, drop unparseable rows, sort,
    drop exact duplicate timestamps (the files carry ~40-49% duplicate rows on MEO),
    and guard against grossly mis-dated rows.
    """
    df = _norm_cols(pd.read_csv(path))
    df["_t"] = pd.to_datetime(df[TIME_COL], format=TIME_FMT, errors="coerce")
    df = df.dropna(subset=["_t"]).sort_values("_t").drop_duplicates("_t")

    # Guard: drop any timestamp more than 60 days from the median (mis-dated rows).
    med = df["_t"].median()
    df = df[(df["_t"] - med).abs() <= pd.Timedelta(days=60)]

    times = df["_t"].to_numpy()
    values = {ch: df[ch].to_numpy(dtype=float) for ch in CHANNELS}
    return times, values


def hours_since(times, t0):
    """Continuous hours of `times` relative to reference datetime `t0`."""
    return (times - t0) / np.timedelta64(1, "h")
