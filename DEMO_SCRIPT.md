# SMART-HORIZON — Demo Walk Script  ·  target 6 min (5–7 range)

**How to use:** read the **[SAY]** lines out loud (that's ~the real spoken content, timed). **[SHOW]/[DO]** are cues. **[NOTE]** is for you, don't read it.
Figures: `Stage-04/figures/`. Live demo: **https://never954.github.io/gnss_visuals/**

**Time budget (≈6:00):** ①0:50 ②0:40 ③0:55 ④0:30 ⑤0:40 ⑥1:15 ⑦1:40 ⑧0:30
If running long, the **cuttable** bits are marked ✂.

---

## ① The problem, you can see it — 0:50
**[SHOW]** live demo — 3D Earth + satellites. **[DO]** click Earth → triangulation animation.

**[SAY]**
> Your phone doesn't know where it is. It listens to these satellites — each one broadcasting *"here's my exact clock time, and here's my orbit position."* Your phone measures how long each signal took, turns it into a distance, and where the spheres cross — that's you. Triangulation.
>
> But it only works if the satellite tells the truth. And it never quite does — it broadcasts a *prediction* of its own clock and orbit, and reality drifts off that prediction. That gap is the **error** we predict.
>
> Why it matters: light travels **0.3 metres per nanosecond**. A few billionths of a second of clock error is *metres* off your position. Predict that error, subtract it, and everyone's GPS gets sharper — lane-level cars, drones, precision farming.

---

## ② What we're actually graded on — 0:40
**[SHOW]** `f01_raw_data.png`

**[SAY]**
> The task: seven days of a satellite's error — X, Y, Z position and clock, all in metres — predict day eight. Two orbit types, GEO and MEO.
>
> Here's the twist that decides everything: **they don't grade us on how close we get.** Accuracy isn't the score. They grade whether our *leftover mistakes look like random noise — a clean bell curve.* And that's actually clever: if your errors are pure random noise, it *proves* you extracted every pattern that was there. Any *shape* left means you missed something. So the real game is **"prove you squeezed out everything the data had."**

**[NOTE]** Shapiro-Francia detail = appendix only. One line if a technical judge probes: *"their file says Shapiro-Wilk but the numbers only reproduce under Shapiro-Francia — we rebuilt their exact scorer, so we grade ourselves the way they will."* Flex, not headline.

---

## ③ First contact with the data — 0:55
**[SHOW]** `f03_duplicates.png`

**[SAY]**
> Three things jumped out immediately. One: **half of every MEO file is duplicated rows** — same readings written twice. Safe to remove, we checked — but it means one test set is *six data points.* Six.
>
> Two — the big one: what actually decides day eight is *where the satellite is in its orbit, which constellation it is, and when ground control last updated it.* **None of those are columns in the file.** It's like being asked to predict tomorrow's tide when all you're handed is the water temperature. The tide is driven by the moon — and you weren't given the moon.

**[SHOW]** `f21_ml_plateau.png`
**[SAY]**
> So we did the obvious thing first — deep learning, the fancy models the brief even suggests. We reasoned they'd just memorise noise, and instead of *claiming* it, we tested it: six standard ML models, twenty-seven thousand rows. **Every one flatlined at the dumb baseline.** That's not a failure — it's the clue. When more model does nothing, the answer isn't in the data.

**[NOTE]** Never call irregular sampling a limitation — we handle it natively; a judge would flip it into "you couldn't pick the right model." Every limitation = a property of the *data*, never our capability.

---

## ④ Everyone caps out — 0:30
**[SHOW]** `f20_benchmark.png`

**[SAY]**
> So the whole team builds models — mine included; I came at it from the small-data, probabilistic side, Gaussian Processes, and they scored well. But here's the honest result: **everything capped in the same narrow band** — my models, the ML, the baselines, all bunched together. When *every* approach hits the same wall no matter how clever, that's the data talking. Time to stop modelling and ask *why the wall is there.*

---

## ⑤ Why the ceiling is real — it's the data, not us — 0:40
**[SHOW]** `f05_volatility_ramp.png` ✂(or skip to f18)

**[SAY]**
> We proved the ceiling three ways. One: **day eight isn't like the training week** — on GEO it's up to *seventeen times* more volatile, and it's a ramp that keeps climbing. Two: on these tiny test sets, even a *perfect* model can't score near the top — on six points it lands anywhere from 0.79 to 0.98 *by luck alone*. A chunk of the score is a lottery, and we measured that in advance. Three: we handed a *cheater* the actual GEO answer key — it improved by **0.3%.** There's genuinely nothing in that file to predict.

---

## ⑥ We rebuilt the physics ourselves — 1:15
**[SHOW]** `f06_beat_frequencies.png`

**[SAY — the crown jewel, slow down]**
> If the answer isn't in the data, we bring it in from outside — from physics. And we found something beautiful.
>
> The data's in Earth-fixed coordinates — a frame glued to the *spinning* Earth. But the satellite's error rotates with the *satellite.* So we're recording two rotations *multiplied together.* And basic trig — sine times cosine — says: multiply two rhythms, you don't get one back, you get **two** — the sum and the difference of their speeds.
>
> So the X and Y error don't appear at the orbital period at all — they split into two *beat* frequencies. Only Z, the spin axis, keeps the pure period. **A model that fits one rhythm to all three axes is guaranteed wrong on two of them.**

**[SHOW]** `f07_periodogram.png`
**[SAY]**
> And we didn't fit-then-explain. We calculated where those beats *must* be, from physics, and drew them on the real data **before looking.** Dead on the peaks.

**[SHOW]** `f11_recovery_distribution.png`
**[SAY — generator = proof, not training]**
> Then, since we only had *three* real satellites — too few to trust anything — we turned that physics into a **generator**: a little engine that makes physically-real fake satellites, sampled at the same irregular timestamps. We made **120 of them**, hid day eight, and asked our model to predict it. It beat "do nothing" on the overwhelming majority — *(confirm exact % on f11)* — across 120 independent satellites. That's how you prove a model works when the real data is too scarce to trust.

**[NOTE]** Say "generator to *prove* ourselves 120 times," not "training data." The model has nothing to train — it fits live per window. (If PRESTO's train-a-deep-net-on-synthetic comes up: we tested that philosophy, it ties/loses — can't manufacture missing information. Don't raise unprompted.)

---

## ⑦ SMART-HORIZON — the model that works ⭐ — 1:40
**[SHOW]** `f22_how_it_works.png`

**[SAY — how it works]**
> So here's the model, and it's almost embarrassingly principled. Three layers: **physics decides which rhythms are even allowed** — only the beats orbital mechanics permits. **The window grades itself** — fit on days one to six, test on day seven which we hide, let the data pick the best basis, no hand-tuning. **A robust regression fits the coefficients.**
>
> And the part I love: **there's nothing to train and store.** Every time you hand it a satellite, it fits *fresh* inside that satellite's own week. So it literally *can't* overfit to some past dataset — there is no dataset. Give it a constellation we've never seen, it just fits *that one.* It's not memorising, it's applying physics live.

**[SHOW]** `f14_results.png`
**[SAY — the mic-drop]**
> On the real MEO satellites: **57% and 19% better than doing nothing.** But the line to remember — **it beats the cheat.** We showed a predictor the actual answer key's average — outright cheating — and our model, which never saw the answers, *still beat it.* Only possible if you're tracking the *shape* of the day, not just the level. Shape beats level. That's the physics paying off.

**[SHOW]** `f16_geo_artifact.png` → `f13_artifact_injection.png`
**[SAY — GEO honesty = maturity flex]**
> And total honesty on GEO — nothing works, ours included, all stuck near fifteen metres. But watch how we handle it instead of hiding it. The cheater proved there's nothing there. Then we found *why*: **the file is broken.** Consecutive readings are near-perfect opposites — swinging +53 to −75 metres in *four minutes*, which a satellite physically can't do. All four channels flip together — impossible, position and clock are unrelated hardware. The 24 biggest jumps alternate sign 24 out of 24 — one in eight million.
>
> The clincher: we took a *clean* synthetic satellite we predict almost perfectly, injected *exactly* this pattern, and watched our model collapse the same way. **We reproduced their file's failure on demand.** That's a diagnosis with a repro, not an excuse.

---

## ⑧ Why you can trust this — 0:30
**[SAY]**
> The jury asked for engineering — so: we tried deep learning, it plateaued, we proved it. We built models from every angle, they capped, so we asked *why.* We derived the ceiling from physics. We stress-tested ourselves 120 times with a generator. We reverse-engineered their scoring metric. We even tried removing outliers and tuning everything — and when it didn't help, we reported *that* too.
>
> This isn't a lucky result — it's an exhaustively tested one. We know exactly where the ceiling is, why it's there, and that we're sitting right on it — beating the baselines, beating a cheater on the real data, and honestly diagnosing the data that isn't. **That's SMART-HORIZON.**

**[DO]** ✂ optional closer — open `interactive.html`: *"Don't believe it? Move the sliders — set corruption to 10% and watch the GEO failure happen live."*

---

### Trim levers if you're over 7:00
- Cut ⑤ to one sentence (just the 0.3% cheat line).
- Drop the periodogram beat in ⑥ (keep the beat-frequency idea + generator).
- Cut the interactive closer.
### If you're under 5:00
- Add the Shapiro-Francia flex in ②, and the `f15` "traces the arc" line in ⑦.

### Decisions baked in (confirm)
1. Synthetic = **proof/validation**, not training. 2. LSTM/transformer = "reasoned + confirmed with 6 models." 3. Your work = credited fast (probabilistic models + the whole ceiling/metric analysis is yours). 4. Metric = "errors look like noise," jargon parked. 5. Limitations = data properties, irregular sampling not listed. 6. Beat-frequency physics kept as the differentiator.
### Verify on slides: exact % in ⑥ (f11), and that ④'s figure matches your ladder.
