# PS-08 Approach Inferences

One section per approach, written **after running it**. Each records what the approach
is, its numbers, and — most importantly — what we *learned* from it. The bare
leaderboard lives in `EVALUATION.md`; this file is the narrative behind the numbers.

> Metric everywhere: **mean Shapiro-Wilk W** of the residual (pred − truth), averaged
> over x/y/z/clock (equal weight). Higher → residuals more Gaussian → better.
> Test sizes: GEO n=69, MEO1 n=6 (unreliable), MEO2 n=18.

---

## B0 — Robust central (`B0_robust_central`)
**What:** predict a single robust constant (median of MAD-clipped training values) per channel.

**Result:** GEO 0.784 · MEO1 0.887 · MEO2 0.809 · **mean 0.827**

**Inferences:**
- This "predict nothing but the centre" floor lands at **0.827** — and every fancier model
  so far is within ~0.01 of it. Concrete proof that **day-ahead forecastability ≈ 0**.
- Because the residual here *is* the centred signal, its W measures the intrinsic
  Gaussianity of the noise. So B0 doubles as a diagnostic of the achievable ceiling.
- Takeaway: the contest is won on **residual shape (outliers)**, not on forecasting skill.

---

## A1 — Robust harmonic (`A1_robust_harmonic`)
**What:** Huber regression on trend (t, t²) + orbital + 24 h harmonics.

**Result:** GEO 0.789 · MEO1 0.884 · MEO2 0.795 · **mean 0.823**

**Inferences:**
- **Slightly *below* the constant floor (0.823 < 0.827).** Adding periodic/trend terms
  injects mild structure into the residual instead of removing it — at the day-8 horizon a
  small period error drifts the phase, so harmonics fit training but mismatch the future.
- Confirms the design rule: **with 46–143 points, lean beats expressive.** Explicit
  periodic terms are not worth their degrees of freedom here.

---

## A2 — Lean Gaussian Process (`A2_gp`)
**What:** RBF (smooth trend) + one learnable periodic + white noise; training values MAD-clipped.

**Result:** GEO 0.789 · MEO1 0.874 · MEO2 0.844 · **mean 0.836 (best so far)**
Per-channel: GEO clk **0.579** (worst), MEO2 y **0.970** (best), MEO2 clk 0.709.

**Inferences:**
- **Best overall, but only +0.009 over a constant.** The GP wins by *not over-committing*
  (its periodic amplitude shrinks when unsupported), not by out-forecasting anything.
- The margin being this thin re-confirms: the ceiling is **outlier-driven, not model-driven**.
- **The wound is the clock channel** — GEO clk 0.579, MEO2 clk 0.709. Heavy-tailed clock
  spikes drag the average. This is precisely what approach **X (space weather)** should target.

---

## A3 — Kalman / state-space (`A3_kalman`)
**What:** continuous-time local linear trend (integrated random walk) Kalman filter, dt-aware
so it handles irregular sampling; process + observation noise fit by MLE.

**Result:** GEO 0.784 · MEO1 0.869 · MEO2 0.801 · **mean 0.818 (worst)**

**Inferences:**
- **Lands *below* the constant floor.** The fitted slope extrapolated over the day-8 gap injects
  trend structure into the residual — the same way A1's harmonics did. On a near-unpredictable
  signal, *any* extrapolation is a liability, not an asset.
- The MLE did shrink the slope (as designed) but not enough to fully collapse to a level. Even a
  principled state-space model cannot beat "predict the centre" here. Third confirmation of the
  central finding.

---

## X — Space weather (`X_spaceweather`) — RULED OUT
**Status:** ❌ not built. Investigated and rejected on evidence.

**Why:**
1. `SW_ReferenceData.xlsx` is **not** space weather — "SW" = **Shapiro-Wilk**. It is a 45-value
   reference residual sample (computed W=0.985, p=0.83): the benchmark our residuals should match
   and a way to validate our own SW code. No space-weather data was ever provided.
2. External Kp confirms the GEO variance-explosion day (**Sept 8, 2025**) was geomagnetically
   **quiet (Kp 1–3)**; the month's first storm was Sept 15 (NOAA SWPC), outside our window.
3. Therefore the outliers are **not** storm-driven — most likely GEO station-keeping maneuvers,
   which are unpredictable from error history. Space weather cannot lift the ceiling here.

**Repurposed value:** use `SW_ReferenceData.xlsx` to validate our Shapiro-W implementation and as
the concrete target (W ≈ 0.98) our residuals are graded against.

---

## C1 — Per-channel ensemble (`C1_ensemble`)
**What:** per channel, hold out the training tail, score each base model's residual normality,
keep the winner, refit on the full series.

**Result:** GEO 0.792 · MEO1 0.899 · MEO2 0.805 · **mean 0.832**

**Inferences:**
- **Does not beat plain GP (0.832 < 0.837).** With ~46 points the held-out-tail Shapiro-W is a
  noisy selection signal — it sometimes picks a model that then does worse on the true test.
  Model *selection* adds variance rather than removing it at this sample size.
- Lesson: in this regime, a single robust model (GP or the constant) is preferable to a
  data-driven selector. A fixed, simple choice out-generalises a clever one.

---

## P1 — Composed pipeline (`P1_composed`)
**What:** the recommended composition (not a vote): Stage 1 robust Theil-Sen detrend →
Stage 2 light MAD outlier clip → Stage 3 lean GP on the residual → Stage 4 validated
Gaussian reporting (`report.py`). Prediction = extrapolated robust trend + GP mean.

**Result:** GEO 0.790 · MEO1 0.907 · MEO2 0.815 · **mean 0.837 (ties A2_gp for best)**

**Inferences:**
- **Composition ties, does not beat, the lean GP.** The Theil-Sen detrend is largely
  redundant with the GP's own RBF trend term — it helps MEO1 (0.907) but costs a little on
  MEO2 (0.815), netting the same 0.837. Confirms again: stacking stages doesn't beat the
  ceiling; it just redistributes where the residual structure lands.
- **Shapiro-W self-check passed:** our implementation scores the benchmark sample at
  W=0.985 (matches the reference), so we are grading residuals exactly as the evaluator will.
- **Per-channel H0 decisions** (α=0.05): MEO1 residuals are statistically normal on all four
  channels (but n=6, low power); GEO and MEO2 reject normality mainly on **clock** (GEO 0.578,
  MEO2 0.710) — the maneuver-driven heavy tails, unchanged and irreducible with these inputs.

---

## Cross-approach inferences (running)
- **Every approach sits in 0.818–0.837 — a 0.02 band.** The forecaster is emphatically *not* the
  differentiator. The barrier is unpredictable outliers (maneuvers), not model quality.
- **Best = A2_gp (0.837); the constant floor (0.827) is within 0.01 of it.** Extrapolators
  (A1 harmonic, A3 Kalman) fall *below* the floor.
- **The real, remaining levers** are all about residual *shape*: outlier/tail treatment, and
  honest handling of the few unpredictable spikes. Not a better trend model, and not
  (given the data) space weather.
- Weakest channels: **clock (GEO 0.58, MEO2 0.71)** and **GEO generally** — the maneuver-driven,
  heavy-tailed ones. This is the ceiling, and it is largely irreducible with the given inputs.
- **Current honest state:** floor ~0.837, ceiling outlier-bound. Beating it needs a new *input*
  (something that predicts the spikes) — which the provided data does not contain.

---

## Clock channel — last effort (negative result)
The clock channel is the worst *and* equally weighted, so it has ~4x leverage on the mean.
We tested clock-specific treatments against the lean-GP baseline (mean W across pairs):

| strategy | mean clock W |
|---|---|
| gp (current) | 0.746 |
| gp heavy-noise | 0.750 |
| theilsen | 0.738 |
| constant | 0.734 |
| arcsinh transform + gp | 0.734 |
| huber (poly+harmonic) | 0.720 |

**Conclusion: nothing legitimately lifts it.** Best treatment beats the GP by 0.004 (noise);
the robust transform and Student-t-style fits do nothing; the extrapolating huber hurts.
The reason is explicit — the score is set by *unpredictable test-truth spikes* we cannot remove:

```
GEO  clock: scored 0.578 -> 0.963 if the 13/69 test-outliers were droppable
MEO2 clock: scored 0.720 -> 0.960 if the  1/18 test-outlier  were droppable
```

Every treatment acts on the training/prediction side; the barrier is on the test side.
**The clock channel, and the ~0.84 overall ceiling, is irreducible with the given inputs.**
The lean GP is already at the wall — keep it simple.

---

## Bias-correction post-process (Priority-2) — tested, does NOT help
Implemented as `BiasCorrected` (estimate bias on a time-ordered training hold-out, subtract
it from predictions). Shapiro-W is unchanged (0.837, shift-invariant, as expected) — but
Priority-2 got slightly *worse*, not better:

| | before | after |
|---|---|---|
| A2_gp \|mean\| | 0.378 m | 0.448 m |
| A2_gp RMSE | 13.217 m | 13.251 m |

**Reason:** the day-8 bias is not predictable from the 7 training days (same
unforecastability as the values themselves), so the hold-out estimate doesn't transfer;
and predictions were already near-unbiased (|mean| ≈ 0.2–0.4 m vs std ≈ 13 m). The wrapper
stays in the registry (`A2_gp_bc`, `P1_composed_bc`) as an option, but is **not recommended**
on this data. Accuracy for the record: pooled MAE 6.04 m, RMSE 13.2 m; clock MAE 15.9 ns.

---

## Not pursued
- **B1 — Synthetic augmentation:** deprioritised by user; revisit only if a new idea needs it.
