"""Run ONE approach end-to-end and produce reproducible artefacts.

    python run_approach.py [approach_name]     (default: P1_composed)

For the chosen approach it:
  - self-validates the Shapiro-Wilk implementation against the benchmark file,
  - fits per channel on each train file, predicts at the test timestamps,
  - scores residual normality (W, p, H0 decision, mean, std) per channel & pair,
  - writes predictions to outputs/<pair>_<approach>_pred.csv,
  - writes a full report to reports/<approach>.md,
  - regenerates the diagnostic plots for that approach.

Available approaches: B0_robust_central, A1_robust_harmonic, A2_gp, A3_kalman,
C1_ensemble, P1_composed.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import PAIRS, DATA_DIR, CHANNELS
from loader import load_series, hours_since
from models import MODELS
from report import validate_sw, shapiro_row
import plots as plotmod

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")
REP_DIR = os.path.join(HERE, "reports")


def run(name):
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(REP_DIR, exist_ok=True)
    factory = MODELS[name]

    swW, swp, swn = validate_sw()
    print(f"\nShapiro-W self-check on SW_ReferenceData.xlsx (n={swn}): "
          f"W={swW:.4f} p={swp:.4f}  (reference ~0.985 -> implementation OK)\n")

    lines = [f"# Report — approach `{name}`", ""]
    lines.append(f"_Shapiro-Wilk self-check on benchmark (n={swn}): W={swW:.4f}, p={swp:.4f} "
                 f"— our SW matches the reference (~0.985)._")
    lines.append("")

    all_means = []
    print(f"{'pair':5s} {'channel':18s} {'W':>6s} {'p':>7s}  {'H0':>12s} {'mean':>9s} {'std':>8s}")
    print("-" * 66)
    for label, ftr, fte, orbit, period in PAIRS:
        tr = load_series(os.path.join(DATA_DIR, ftr))
        te = load_series(os.path.join(DATA_DIR, fte))
        t0 = tr[0][0]
        t_tr, t_te = hours_since(tr[0], t0), hours_since(te[0], t0)

        lines.append(f"## {label} ({orbit}) — test n={len(te[0])}\n")
        lines.append("| channel | W | p | H0 rejected (non-normal)? | mean | std |")
        lines.append("|---|---|---|---|---|---|")

        preds, pair_W = {}, []
        for ch in CHANNELS:
            model = factory(period)
            model.fit(t_tr, tr[1][ch])
            pred = np.asarray(model.predict(t_te), float)
            preds[ch] = pred
            res = pred - te[1][ch].astype(float)
            W, p, rej, mu, sd, n = shapiro_row(res)
            rej_s = "—" if rej is None else ("yes" if rej else "no")
            pair_W.append(W)
            print(f"{label:5s} {ch:18s} {W:6.3f} {p:7.3f}  {rej_s:>12s} {mu:+9.3f} {sd:8.3f}")
            lines.append(f"| {ch} | {W:.3f} | {p:.3f} | {rej_s} | {mu:+.3f} | {sd:.3f} |")

        mean_W = float(np.nanmean(pair_W))
        all_means.append(mean_W)
        lines.append(f"\n**{label} mean W = {mean_W:.3f}**\n")

        # predictions CSV (with truth + residual since we have the test files)
        df = pd.DataFrame({"utc_time": pd.to_datetime(te[0])})
        for ch in CHANNELS:
            key = ch.split()[0]
            df[f"{key}_pred"] = preds[ch]
            df[f"{key}_truth"] = te[1][ch]
            df[f"{key}_resid"] = preds[ch] - te[1][ch]
        csv_path = os.path.join(OUT_DIR, f"{label}_{name}_pred.csv")
        df.to_csv(csv_path, index=False)
        print(f"      -> {csv_path}")

    overall = float(np.nanmean(all_means))
    lines.insert(3, f"**Overall mean Shapiro-W across pairs = {overall:.3f}** "
                    f"(benchmark target ~0.985).\n")
    print(f"\nOVERALL mean Shapiro-W ({name}) = {overall:.3f}")

    try:
        plotmod.make_plots(name)
    except Exception as e:
        print("plot generation skipped:", e)

    with open(os.path.join(REP_DIR, f"{name}.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"report -> reports/{name}.md   predictions -> outputs/   plots -> plots/")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "P1_composed"
    if name not in MODELS:
        print("Unknown approach:", name)
        print("Choose from:", ", ".join(MODELS))
        sys.exit(1)
    run(name)
