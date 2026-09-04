"""Run every registered model on every train->test pair, print a Shapiro-Wilk
leaderboard, and rewrite the auto-generated results block in EVALUATION.md.

Usage:  python run.py
"""
import os
from datetime import date
import numpy as np

from config import PAIRS, DATA_DIR, CHANNELS
from loader import load_series
from models import MODELS
from evaluate import evaluate_pair

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(HERE, "EVALUATION.md")
START, END = "<!--RESULTS_START-->", "<!--RESULTS_END-->"
CH_SHORT = {"x_error (m)": "x", "y_error (m)": "y", "z_error (m)": "z",
            "satclockerror (m)": "clk"}


def main():
    # Load each pair once.
    pairs = []
    for label, ftr, fte, orbit, period in PAIRS:
        train = load_series(os.path.join(DATA_DIR, ftr))
        test = load_series(os.path.join(DATA_DIR, fte))
        pairs.append((label, orbit, period, train, test, len(test[0])))

    # model -> {pair_label: mean_W}, plus per-channel for the appendix.
    results = {name: {} for name in MODELS}
    per_channel_dump = {name: {} for name in MODELS}
    for name, factory in MODELS.items():
        for label, orbit, period, train, test, n_te in pairs:
            meanW, per_ch = evaluate_pair(factory, period, train, test)
            results[name][label] = meanW
            per_channel_dump[name][label] = per_ch

    labels = [p[0] for p in pairs]
    # Console print
    print(f"\n{'approach':22s} " + " ".join(f"{l:>7s}" for l in labels) + "   mean")
    print("-" * (22 + 8 * len(labels) + 7))
    ranking = []
    for name in MODELS:
        row = [results[name][l] for l in labels]
        overall = np.nanmean(row)
        ranking.append((overall, name))
        print(f"{name:22s} " + " ".join(f"{v:7.3f}" for v in row) + f"   {overall:6.3f}")

    # ---- write ledger block ----
    lines = []
    lines.append(f"_Last run: {date.today().isoformat()} · metric: mean Shapiro-Wilk W "
                 f"of residual (pred − truth), averaged over x/y/z/clock. Higher = better._\n")
    header = "| Approach | " + " | ".join(f"{l} W" for l in labels) + " | **Mean W** |"
    sep = "|" + "---|" * (len(labels) + 2)
    lines.append(header)
    lines.append(sep)
    for overall, name in sorted(ranking, reverse=True):
        cells = " | ".join(f"{results[name][l]:.3f}" for l in labels)
        lines.append(f"| `{name}` | {cells} | **{overall:.3f}** |")
    lines.append("")
    lines.append(f"Test points per pair: " +
                 ", ".join(f"{p[0]}={p[5]}" for p in pairs) +
                 " (small n — treat single-channel W cautiously).")
    lines.append("")
    # Per-channel appendix for the best model
    best_name = sorted(ranking, reverse=True)[0][1]
    lines.append(f"**Per-channel detail — best approach (`{best_name}`):**\n")
    lines.append("| Pair | " + " | ".join(CH_SHORT[c] for c in CHANNELS) + " |")
    lines.append("|" + "---|" * (len(CHANNELS) + 1))
    for label in labels:
        pc = per_channel_dump[best_name][label]
        cells = " | ".join(f"{pc[c][0]:.3f}" for c in CHANNELS)
        lines.append(f"| {label} | {cells} |")
    block = "\n".join(lines)

    with open(LEDGER, "r") as f:
        text = f.read()
    pre, _, rest = text.partition(START)
    _, _, post = rest.partition(END)
    new = f"{pre}{START}\n{block}\n{END}{post}"
    with open(LEDGER, "w") as f:
        f.write(new)
    print(f"\nLedger updated -> {LEDGER}")


if __name__ == "__main__":
    main()
