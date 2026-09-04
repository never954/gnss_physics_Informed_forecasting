"""Residual histograms with a fitted Gaussian overlay — the visual companion to the
Shapiro-Wilk statistic. For each channel, residuals (pred - truth) are standardised within
each pair (z = (r-mean)/std) and pooled, then compared to the standard normal N(0,1).

Usage:  python gaussianity_curves.py [model_name]   (default: A2_gp)
Output: plots/gaussianity_<model>.png
"""
import os
import sys
import warnings
import numpy as np

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import shapiro, norm

from config import PAIRS, DATA_DIR, CHANNELS
from loader import load_series, hours_since
from models import MODELS

HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(HERE, "plots")
SHORT = {"x_error (m)": "x", "y_error (m)": "y", "z_error (m)": "z", "satclockerror (m)": "clock"}


def make(model_name="A2_gp"):
    os.makedirs(PLOTS, exist_ok=True)
    factory = MODELS[model_name]
    pooled = {ch: [] for ch in CHANNELS}
    for label, ftr, fte, orb, per in PAIRS:
        tr = load_series(os.path.join(DATA_DIR, ftr))
        te = load_series(os.path.join(DATA_DIR, fte))
        t0 = tr[0][0]
        ttr, tte = hours_since(tr[0], t0), hours_since(te[0], t0)
        for ch in CHANNELS:
            m = factory(per)
            m.fit(ttr, tr[1][ch].astype(float))
            res = np.asarray(m.predict(tte), float) - te[1][ch].astype(float)
            if res.std() > 1e-12:
                pooled[ch].extend(list((res - res.mean()) / res.std()))  # standardise then pool

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    grid = np.linspace(-4, 4, 200)
    for ax, ch in zip(axes.ravel(), CHANNELS):
        z = np.array(pooled[ch])
        W = shapiro(z)[0] if len(z) >= 3 else float("nan")
        ax.hist(z, bins=18, density=True, color="#0b7d8c", alpha=0.55,
                edgecolor="white", linewidth=0.6, label="standardised residuals")
        ax.plot(grid, norm.pdf(grid), color="#c0392b", lw=2.2, label="ideal Gaussian N(0,1)")
        ax.set_title(f"{SHORT[ch]}   ·   Shapiro-W = {W:.3f}   (n={len(z)})", fontsize=11)
        ax.set_xlabel("standardised residual (σ)", fontsize=9)
        ax.set_ylabel("density", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8)
    fig.suptitle(f"Residual Gaussianity — model {model_name}\n"
                 f"(bars = our residuals, red = perfect Gaussian; closer = better, W→1)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(PLOTS, f"gaussianity_{model_name}.png")
    fig.savefig(out, dpi=115)
    plt.close(fig)
    print("wrote", out)
    return out


if __name__ == "__main__":
    make(sys.argv[1] if len(sys.argv) > 1 else "A2_gp")
