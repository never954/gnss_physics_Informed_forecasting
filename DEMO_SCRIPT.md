# SMART-HORIZON — Demo Walk Script  ·  target 6 min (5–7 range)

**How to use:** read the **[SAY]** lines out loud (that's ~the real spoken content, timed). **[SHOW]/[DO]** are cues. **[NOTE]** is for you, don't read it.
Figures: `Stage-04/figures/`. Live demo: **https://never954.github.io/gnss_visuals/**

**Time budget (≈6:00):** ①0:50 ②0:40 ③0:55 ④0:30 ⑤0:40 ⑥1:15 ⑦1:40 ⑧0:30
If running long, the **cuttable** bits are marked ✂.

---

## ① The problem, made visible — 0:50
**[SHOW]** live demo — 3D Earth + satellites. **[DO]** click Earth → triangulation animation.

**[SAY]**
> Your phone doesn't know where it is. It listens to these satellites — each one broadcasting *"here's my exact clock time, and here's my orbit position."* Your phone measures how long each signal took, turns it into a distance, and where the spheres cross — that's you. Triangulation.
>
> But it only works if the satellite tells the truth. And it never quite does — it broadcasts a *prediction* of its own clock and orbit, and reality drifts off that prediction. That gap is the **error** we predict.
>
> Why it matters: light travels **0.3 metres per nanosecond**. A few billionths of a second of clock error is *metres* off your position. Predict that error, subtract it, and everyone's GPS gets sharper — lane-level cars, drones, precision farming.

---

## ② What we're graded on — 0:40
**[SHOW]** `f01_raw_data.png`

**[SAY]**
> The task: seven days of a satellite's error — X, Y, Z position and clock, in metres — predict day eight. Two orbit types, GEO and MEO.
>
> One thing about the scoring is worth being clear on, because it isn't the obvious choice: they don't primarily grade *accuracy* — how close we get. They grade whether the *residual* — what's left after our prediction — follows a normal distribution, a bell curve. The reasoning is sound: if the leftover error is just random noise, the model has captured the predictable structure; if the residual still has shape — a trend, a cycle — there's structure it missed. So the metric tests whether we removed the predictable part, not whether the noise floor is small.

**[NOTE]** Shapiro-Francia detail = appendix only. One line if a technical judge probes: *"their file says Shapiro-Wilk but the numbers only reproduce under Shapiro-Francia — we rebuilt their exact scorer, so we grade ourselves the way they will."*

---

## ③ First look at the data — 0:55
**[SHOW]** `f03_duplicates.png`

**[SAY]**
> Three things stood out immediately. One: **about half of every MEO file is duplicated rows** — the same readings written twice. Safe to remove, we verified that — but it means one test set is only *six points*.
>
> Two, and this is the important one: what actually determines day eight is where the satellite is in its orbit, which constellation it is, and when ground control last updated it. **None of those are columns in the file.** It's a bit like being asked to predict the tide from water temperature — the thing that drives it isn't in your inputs.

**[SHOW]** `f21_ml_plateau.png`
**[SAY]**
> So we tried the obvious thing first — the deep-learning models the brief suggests. We expected them to overfit on this little data, but rather than assume it, we ran six standard models on twenty-seven thousand rows. **All six landed at essentially the trivial baseline** — no better than predicting the weekly average. That's informative: if adding model capacity doesn't help, the limit isn't the model, it's that the inputs don't determine the output.

**[NOTE]** Never call irregular sampling a limitation — we handle it natively; a judge would flip it into "you couldn't pick the right model." Every limitation = a property of the *data*, never our capability.

---

## ④ Every approach caps out — 0:30
**[SHOW]** `f20_benchmark.png`

**[SAY]**
> The whole team built models — mine came from the small-data, probabilistic side, Gaussian Processes, and they scored competitively. But the honest result is that **everything converged to the same narrow band** — my models, the standard ML, the baselines. When independent approaches all plateau at the same level, that points to a limit in the data rather than the models. So the useful question became *why* that limit exists.

---

## ⑤ Why the ceiling is real — it's the data, not the model — 0:40
**[SHOW]** `f05_volatility_ramp.png` ✂(or skip to f18)

**[SAY]**
> We checked that ceiling three ways. One: **day eight isn't drawn from the same distribution as the training week** — on GEO it's up to seventeen times more volatile, and it's a steady ramp, not a one-off spike. Two: at these test-set sizes even a *perfect* model can't score near the top — on six points a flawless predictor lands anywhere from 0.79 to 0.98 by chance, so part of the score is genuinely luck. Three: we gave a predictor the actual GEO answer key's average — which no real model could use — and it improved on doing nothing by **0.3%**. There's effectively no predictable signal in that file.

---

## ⑥ Bringing physics in from outside the data — 1:15
**[SHOW]** `f06_beat_frequencies.png`

**[SAY]**
> If the information that decides day eight isn't in the data, it has to come from outside it — from physics. The key result is this.
>
> The data is in Earth-fixed coordinates — a frame attached to the rotating Earth. But the satellite's error rotates with the *satellite*. So what's recorded is the two rotations multiplied together, and there's a standard trig identity — sine times cosine — where multiplying two rhythms produces two new ones, at the sum and the difference of their frequencies. So the X and Y error don't appear at the orbital period; they split into two *beat* frequencies. Only Z, the rotation axis, keeps the orbital period. A model that fits one rhythm to all three axes is therefore wrong on two of them.

**[SHOW]** `f07_periodogram.png`
**[SAY]**
> And this is derived, not fitted after the fact: we computed where those beat frequencies must be, marked them on the real data before looking, and they line up with the peaks.

**[SHOW]** `f11_recovery_distribution.png`
**[SAY]**
> We only had three real satellites, which is too few to trust any result on. So we used the same physics to build a generator — it produces physically consistent synthetic satellites, sampled at the real files' irregular timestamps. We generated 120, held out day eight on each, and the model recovered it, beating the do-nothing baseline on the large majority *(confirm exact % on f11)*. That's how we validated it despite the scarce real data.

**[NOTE]** Say "generator to *validate* across 120 cases," not "training data." The model has nothing to train — it fits live per window. (If PRESTO's train-a-deep-net-on-synthetic comes up: we tested that philosophy, it ties/loses — can't manufacture missing information. Don't raise unprompted.)

---

## ⑦ SMART-HORIZON — the model — 1:40
**[SHOW]** `f22_how_it_works.png`

**[SAY]**
> The model itself is deliberately simple, in three layers. **Physics fixes which rhythms are admissible** — only the beat frequencies orbital mechanics allows. **The window selects its own basis** — we fit on days one to six, score on a held-out day seven, and let the data choose. **A regularised regression fits the coefficients.**
>
> One consequence worth noting: there's no stored trained model. It fits each satellite's own week at prediction time, so there's no training set it can be overfit to — which is also why it should transfer to constellations we haven't seen.

**[SHOW]** `f14_results.png`
**[SAY]**
> On the real MEO satellites it's 57% and 19% better than doing nothing. A stronger check: it also beats a predictor that was handed the test set's own average — something no legitimate model can use. That's only possible by tracking the *shape* of the error through the day, not just its average level.

**[SHOW]** `f16_geo_artifact.png` → `f13_artifact_injection.png`
**[SAY]**
> On GEO, nothing works — ours included; everything sits near fifteen metres. But we can show that's the data, not the model. The answer-key predictor improved by 0.3%, and when we looked at why, the file turns out to be non-physical: consecutive readings are near-exact opposites, swinging +53 to −75 metres in four minutes, which an orbiting satellite can't do; all four channels flip sign together, though position and clock come from unrelated hardware; and the 24 largest jumps alternate sign 24 times out of 24. To confirm it's the artefact and not us, we injected that same pattern into a clean synthetic satellite we predict well, and the model failed in the same way. So the corruption claim is reproducible, not just inferred.

---

## ⑧ Why the result is trustworthy — 0:30
**[SAY]**
> To summarise the engineering path: we tried deep learning and showed it plateaus; we built models from several directions and they capped at the same level, so we investigated why; we derived that ceiling from the physics and from the data itself; we validated on 120 generated cases; we reimplemented the scoring metric to grade ourselves correctly; and we tested the usual refinements like outlier removal, reporting them even where they didn't help.
>
> So the result rests on testing at each step rather than on a single model that happened to work: we can say where the ceiling is, why it's there, and that the model sits at it — beating the baselines and the answer-key predictor on the real data, and correctly diagnosing the data that has no signal. That's SMART-HORIZON.

**[DO]** ✂ optional closer — open `interactive.html`: *"If you want to check it directly — this sandbox runs the same algorithm. Set corruption to 10% and the GEO failure reproduces live."*

---

### Trim levers if you're over 7:00
- Cut ⑤ to one sentence (just the 0.3% answer-key line).
- Drop the periodogram beat in ⑥ (keep the beat-frequency idea + generator).
- Cut the interactive closer.
### If you're under 5:00
- Add the Shapiro-Francia line in ②, and an `f15` line in ⑦ (the model traces the arc while the average sits flat).

### Decisions baked in (confirm)
1. Synthetic = **validation**, not training. 2. LSTM/transformer = "expected overfit, confirmed with six models." 3. Your work credited fast (probabilistic models + the ceiling/metric analysis). 4. Metric = residual normality, in plain terms; Shapiro-Francia parked. 5. Limitations = data properties; irregular sampling not listed. 6. Beat-frequency physics kept.
### Verify on slides: exact % in ⑥ (f11), and that ④'s figure matches your ladder.
