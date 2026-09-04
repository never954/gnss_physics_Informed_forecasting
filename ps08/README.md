# ps08 — reality-check pipeline on the *actual* competition data

The main `src/` pipeline is the full physics-decomposition system (GP + Bootstrap-MC +
Student-t + Matérn), built and tuned on a **self-constructed, regularly-sampled,
multi-constellation NASA dataset** in nanoseconds, optimised for **accuracy (MAE/RMSE/CRPS)**.

`ps08/` is different by necessity: it targets the data the competition **actually** provides
(`DATA_PS-08`), which turned out to be a different problem:

| | main `src/` pipeline | this `ps08/` |
|---|---|---|
| Data | self-built NASA (78-col, multi-constellation) | given files: 1 GEO + 2 MEO |
| Sampling | regular 15-min | **irregular** (10 min – 2 h, mixed) |
| Target | SISE in **ns** (computed) | **x/y/z/clock errors in metres**, given directly |
| Data size | large | **46–143 unique points/sat** (~40–49% duplicate rows) |
| Metric | accuracy (MAE/RMSE/CRPS) | **Shapiro-Wilk normality of residuals** |

## What we learned (see `INFERENCES.md`)
- Day-ahead forecastability is **≈ 0**; a lean GP barely beats a constant.
- The score is set by **unpredictable, maneuver-driven outliers** (GEO clock worst), not by
  trend modelling. Ceiling ≈ **0.84** mean Shapiro-W; every model sits in a 0.02 band.
- Our Shapiro-W implementation is **validated against `SW_ReferenceData.xlsx`** (W = 0.985 ✓).

## Approaches (leaderboard in `EVALUATION.md`)
`B0` robust-central · `A1` robust-harmonic · `A2` lean GP (best, 0.837) ·
`A3` continuous Kalman · `C1` per-channel ensemble · `P1` composed (detrend→clip→GP, ties best).
Space-weather (`X`) was investigated and **ruled out** — `SW_ReferenceData.xlsx` is the
Shapiro-Wilk benchmark, not space weather, and the outlier day was geomagnetically quiet.

## Run it
```bash
cd ps08
python3 run.py                    # rebuild the leaderboard (EVALUATION.md)
python3 run_approach.py A2_gp     # one approach end-to-end -> outputs/, reports/, plots/
```
Data resolves from `$PS08_DATA_DIR`, else the bundled `ps08/data/`, else a local fallback.
