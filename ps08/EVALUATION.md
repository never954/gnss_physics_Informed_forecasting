# PS-08 Evaluation Ledger

The running cross-reference for every approach we try. The results table below is
**auto-generated** by `run.py` — do not edit between the markers. Everything else
(notes, decisions, approach log) is hand-maintained.

**Problem in one line:** predict day-8 x/y/z/clock errors (meters, ECEF) at arbitrary
irregular timestamps from ~46–143 real points per satellite. **Scored on Shapiro-Wilk
normality of the residual** (pred − truth), x/y/z/clock equally weighted. Higher W → better.

---

## Leaderboard (auto-generated)

<!--RESULTS_START-->
_Last run: 2026-09-04 · metric: mean Shapiro-Wilk W of residual (pred − truth), averaged over x/y/z/clock. Higher = better._

| Approach | GEO W | MEO1 W | MEO2 W | **Mean W** |
|---|---|---|---|---|
| `P1_composed` | 0.790 | 0.907 | 0.815 | **0.837** |
| `A2_gp` | 0.791 | 0.890 | 0.829 | **0.837** |
| `C1_ensemble` | 0.792 | 0.899 | 0.805 | **0.832** |
| `B0_robust_central` | 0.784 | 0.887 | 0.809 | **0.827** |
| `A1_robust_harmonic` | 0.789 | 0.884 | 0.795 | **0.823** |
| `A3_kalman` | 0.784 | 0.869 | 0.801 | **0.818** |

Test points per pair: GEO=69, MEO1=6, MEO2=18 (small n — treat single-channel W cautiously).

**Per-channel detail — best approach (`P1_composed`):**

| Pair | x | y | z | clk |
|---|---|---|---|---|
| GEO | 0.870 | 0.821 | 0.890 | 0.578 |
| MEO1 | 0.856 | 0.981 | 0.844 | 0.948 |
| MEO2 | 0.856 | 0.862 | 0.832 | 0.710 |
<!--RESULTS_END-->

---

## What we know (grounded findings)

- **Data scarcity is real:** MEO files are ~40–49% duplicate rows → GEO ≈142, MEO-sat1 ≈46,
  MEO-sat2 ≈143 unique points. One satellite per file, no ID column, irregular sampling.
- **Day-ahead forecastability ≈ 0:** a physics-kernel GP collapses to ~constant; residual
  spread ≈ raw-signal spread. Elaborate forecasting buys little.
- **Outliers set the score:** removing the sparse 15–30% heavy points lifts W from ~0.78 to
  ~0.95–0.98. Excess kurtosis +3 to +18. Outlier treatment is the dominant lever.
- **GEO vs MEO:** GEO is noise-like (lag-1 AC −0.2 to −0.5, tens-of-meters) and the weaker
  half; MEO is smoother (lag-1 AC +0.4 to +0.98, sub-meter). Same pipeline, different weight.
- **Ceiling caveat:** W is capped by *unpredictable* test-truth outliers we cannot remove.
  The only lever that breaks that ceiling is predicting the spikes (space weather, approach X).

---

## Approach log

| # | Approach | Status | Idea in one line |
|---|---|---|---|
| B0 | `B0_robust_central` | ✅ implemented | Predict robust constant level — the floor. |
| A1 | `A1_robust_harmonic` | ✅ implemented | Huber trend + orbital + daily harmonics. |
| A2 | `A2_gp` | ✅ implemented | Lean GP: RBF + learnable period + white noise. |
| A3 | `A3_kalman` | ✅ implemented | Continuous-time local level+trend Kalman (dt-aware). |
| X  | `X_spaceweather` | ❌ ruled out | No SW data provided ("SW"=Shapiro-Wilk); Sept 8 was Kp-quiet. |
| B1 | `B1_synthetic_ml` | ⏸ skipped | PRESTO-style synthetic augmentation (deprioritised by user). |
| C1 | `C1_ensemble` | ✅ implemented | Per-channel pick the highest held-out W. |
| P1 | `P1_composed` | ✅ implemented | Robust detrend → light clip → lean GP (composition). |

**How to reproduce (each approach is its own runnable):**
`python run.py` rebuilds this leaderboard; `python run_approach.py <name>` runs one approach
end-to-end → predictions in `outputs/`, a report in `reports/<name>.md`, plots in `plots/`.
Approaches: `B0_robust_central A1_robust_harmonic A2_gp A3_kalman C1_ensemble P1_composed`.
Our Shapiro-W is self-validated against `SW_ReferenceData.xlsx` (W=0.985 ✓).

## Decisions & notes

- Metric harness scores directly on the provided train→test pairs (the real day-8 truth).
- Models are per-series / per-channel and continuous-time → robust to irregular timestamps
  and to the evaluator using different satellites.
- Reference companion doc (design + rationale): the "PS-08 Approach Ledger" artifact.
