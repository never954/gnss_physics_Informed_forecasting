"""Evaluation harness — the competition metric.

For a given model and a train->test pair: fit per channel on the train series,
predict at the exact test timestamps, form the residual (pred - truth), and score
Shapiro-Wilk W (+ p, mean, std) per channel. The headline is the mean W across the
four equally-weighted channels.
"""
import warnings
import numpy as np
from scipy.stats import shapiro

from config import CHANNELS
from loader import hours_since


def evaluate_pair(model_factory, period, train, test):
    """Returns (mean_W, per_channel dict). per_channel[ch] = (W, p, mean, std, n)."""
    times_tr, vals_tr = train
    times_te, vals_te = test
    t0 = times_tr[0]
    t_tr = hours_since(times_tr, t0)
    t_te = hours_since(times_te, t0)

    per_channel = {}
    Ws = []
    for ch in CHANNELS:
        try:
            model = model_factory(period)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(t_tr, vals_tr[ch])
                pred = model.predict(t_te)
            res = np.asarray(pred, float) - np.asarray(vals_te[ch], float)
            if len(res) >= 3:
                W, p = shapiro(res)
            else:
                W, p = np.nan, np.nan
            per_channel[ch] = (W, p, float(np.mean(res)), float(np.std(res)), len(res))
            Ws.append(W)
        except Exception as e:  # keep one channel's failure from killing the run
            per_channel[ch] = (np.nan, np.nan, np.nan, np.nan, 0)
            Ws.append(np.nan)
    return float(np.nanmean(Ws)) if np.any(~np.isnan(Ws)) else np.nan, per_channel
