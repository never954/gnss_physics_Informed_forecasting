"""Visualisation for understanding the data and each model's behaviour.

For every train->test pair it draws one figure per model with, for each channel:
  left  — the time series (train inliers, train outliers, test truth, prediction);
  right — the residual normal-QQ plot with its Shapiro-W (the scored quantity).

Usage:  python plots.py [model_name]   (default: A2_gp)
Outputs PNGs into ps08/plots/.
"""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from config import PAIRS, DATA_DIR, CHANNELS
from loader import load_series, hours_since
from models import MODELS
from outliers import outlier_mask

HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(HERE, "plots")
CH_SHORT = {"x_error (m)": "x", "y_error (m)": "y", "z_error (m)": "z",
            "satclockerror (m)": "clock"}

C_TRAIN, C_OUT, C_TEST, C_PRED = "#8a97a3", "#c0392b", "#1f6aa5", "#0b7d8c"


def make_plots(model_name="A2_gp"):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    factory = MODELS[model_name]
    written = []
    for label, ftr, fte, orbit, period in PAIRS:
        times_tr, vals_tr = load_series(os.path.join(DATA_DIR, ftr))
        times_te, vals_te = load_series(os.path.join(DATA_DIR, fte))
        t0 = times_tr[0]
        t_tr, t_te = hours_since(times_tr, t0), hours_since(times_te, t0)
        grid = np.linspace(float(t_tr.min()), float(t_te.max()), 400)

        fig, axes = plt.subplots(4, 2, figsize=(12, 13))
        for i, ch in enumerate(CHANNELS):
            model = factory(period)
            try:
                model.fit(t_tr, vals_tr[ch])
                pred_grid = model.predict(grid)
                pred_te = model.predict(t_te)
            except Exception:
                pred_grid = np.full_like(grid, np.nan)
                pred_te = np.full_like(t_te, np.nan)

            mask = outlier_mask(vals_tr[ch])
            axL = axes[i, 0]
            axL.scatter(t_tr[mask], vals_tr[ch][mask], s=13, c=C_TRAIN, label="train")
            axL.scatter(t_tr[~mask], vals_tr[ch][~mask], s=30, c=C_OUT,
                        marker="x", linewidths=1.4, label="train outlier")
            axL.scatter(t_te, vals_te[ch], s=18, c=C_TEST, label="test truth", zorder=5)
            axL.plot(grid, pred_grid, c=C_PRED, lw=1.7, label="prediction")
            axL.axvline(float(t_tr.max()), ls=":", c="#aaa", lw=1)
            axL.set_title(f"{label} · {CH_SHORT[ch]}", fontsize=10)
            axL.set_xlabel("hours since train start", fontsize=8)
            axL.tick_params(labelsize=7)
            if i == 0:
                axL.legend(fontsize=7, loc="best")

            axR = axes[i, 1]
            res = np.asarray(pred_te, float) - np.asarray(vals_te[ch], float)
            res = res[np.isfinite(res)]
            if len(res) >= 3:
                stats.probplot(res, dist="norm", plot=axR)
                W = stats.shapiro(res)[0]
                axR.set_title(f"residual QQ · {CH_SHORT[ch]} · W={W:.3f}", fontsize=10)
            else:
                axR.set_title(f"residual QQ · {CH_SHORT[ch]} · n<3", fontsize=10)
            axR.set_xlabel("theoretical quantiles", fontsize=8)
            axR.set_ylabel("ordered residual", fontsize=8)
            axR.tick_params(labelsize=7)

        fig.suptitle(f"{label}  —  model: {model_name}", fontsize=13, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.985])
        out = os.path.join(PLOTS_DIR, f"{label}_{model_name}.png")
        fig.savefig(out, dpi=110)
        plt.close(fig)
        written.append(out)
        print("wrote", out)
    return written


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "A2_gp"
    make_plots(name)
