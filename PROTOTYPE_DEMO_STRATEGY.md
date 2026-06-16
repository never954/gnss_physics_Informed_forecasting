# GNSS Error Prediction Prototype Demo Strategy

## Objective

Demonstrate an end-to-end AI/ML prototype that predicts time-varying GNSS satellite clock and ephemeris error buildup from seven days of historical data and forecasts the eighth day at 15-minute intervals.

The prototype addresses the challenge goal:

> Develop AI/ML based models to predict time-varying patterns of the error buildup between uploaded and modelled values of both satellite clock and ephemeris parameters of navigation satellites.

## Problem Context

GNSS positioning accuracy is limited by satellite clock bias and ephemeris/orbit prediction errors. These errors evolve after broadcast upload and can degrade positioning and timing reliability.

The challenge provides seven days of satellite error data for multiple GNSS constellations and requires prediction for an unseen eighth day. Evaluation focuses on forecast accuracy at:

- 15 minutes
- 30 minutes
- 1 hour
- 2 hours
- 24 hours

The error distribution is also evaluated for closeness to a normal distribution.

## Prototype Summary

This prototype builds a complete forecasting pipeline:

1. Ingest multi-constellation GNSS CSV files.
2. Convert broadcast/merged GNSS data into per-satellite SISE time series.
3. Classify each satellite by constellation, orbit type, and reset behavior.
4. Route each satellite to an appropriate probabilistic model.
5. Predict Day 8 errors at 15-minute intervals.
6. Evaluate forecast accuracy, uncertainty calibration, and Gaussianity diagnostics.

## Demo Command

Run the complete pipeline and evaluation with:

```bash
python src/pipeline.py --competition-data data/raw/train/ --output outputs/ 2>&1 | tee outputs/pipeline_run.log && python src/evaluate.py --predictions outputs/submission.csv --actual-data data/raw/train/2026_008.csv --output outputs/evaluation_report.csv
```

## Data Usage

Training input:

```text
data/raw/train/2026_001.csv
data/raw/train/2026_002.csv
...
data/raw/train/2026_007.csv
```

Held-out validation target:

```text
data/raw/train/2026_008.csv
```

The first seven days are used for training. The eighth day is not used during model fitting and is used only for evaluation.

## Pipeline Stages

### 1. Competition Data Preprocessing

The raw competition CSVs contain 78 GNSS columns with broadcast ephemeris, clock terms, constellation metadata, and available IGS truth.

The preprocessing layer converts this into per-satellite 15-minute SISE/error time series:

```text
78-column GNSS CSVs
        -> per-satellite SISE CSVs
        -> data/processed/sise_intermediate/
```

This lets the downstream ML pipeline work with a consistent time-series format across GPS, Galileo, BeiDou, GLONASS, QZSS, NavIC, and SBAS.

### 2. Satellite Classification

Each satellite is classified by:

- Satellite ID and constellation
- Orbit type: MEO or GEO/GSO
- Reset behavior: clean, regular sawtooth, or irregular
- Recommended model type

The classification output is saved to:

```text
outputs/classification_log.csv
```

### 3. Model Routing

The prototype uses different probabilistic models depending on satellite behavior:

| Satellite Behavior | Model Used | Purpose |
|---|---|---|
| Clean / smooth | Gaussian Process | Learns clock drift and orbital/solar periodicity |
| Regular reset pattern | Bootstrap Monte Carlo | Generates possible future reset trajectories |
| Irregular reset pattern | Student-t Process | Robust to outliers and heavy-tailed jumps |
| Model failure / unstable case | Matérn fallback | Ensures prediction coverage |

This demonstrates an adaptive AI/ML approach rather than a single model applied blindly to every satellite.

### 4. Prediction Output

Predictions are saved to:

```text
outputs/submission.csv
```

Each row contains:

```text
sat_id, timestamp, mean_ns, std_ns, horizon_min
```

The model predicts 96 future points per satellite:

```text
96 points = 24 hours x 4 samples/hour
```

### 5. Evaluation

The evaluation script compares predicted Day 8 values against actual Day 8 processed values.

Full evaluation report:

```text
outputs/evaluation_report.csv
```

Metrics include:

- MAE: mean absolute error
- RMSE: root mean squared error
- MBE: mean bias error
- CRPS: probabilistic forecast score
- sigma-ratio: uncertainty calibration
- coverage within 1-sigma and 2-sigma intervals
- skewness and kurtosis diagnostics

## Current Prototype Result

Recent evaluation summary:

```text
MAE       = 91.179 ns
RMSE      = 422.840 ns
MBE       = 46.934 ns
CRPS      = 63.3724
Cov +/-1s = 74.5%
Cov +/-2s = 92.9%
```

Equivalent average range error:

```text
91.179 ns x 0.2998 m/ns ~= 27 meters
```

Horizon-wise performance:

```text
15 min   MAE ~= 53 ns
30 min   MAE ~= 55 ns
1 hour   MAE ~= 64 ns
2 hours  MAE ~= 65 ns
24 hours MAE ~= 135 ns
```

Interpretation:

- Short-horizon forecasts are stronger.
- Error grows toward the 24-hour horizon, as expected.
- The model provides uncertainty estimates in addition to point predictions.
- Most satellites are covered across multiple GNSS constellations.
- A small number of difficult satellites dominate the high RMSE.

## Demo Talking Points

Use this short explanation during the demo:

> This prototype demonstrates an end-to-end AI/ML pipeline for predicting GNSS satellite clock and ephemeris error buildup. It ingests seven days of multi-constellation broadcast data, transforms it into satellite-level time series, detects reset and upload behavior, selects a probabilistic forecasting model per satellite, and predicts the eighth day at 15-minute intervals. The system evaluates performance across 15-minute, 30-minute, 1-hour, 2-hour, and 24-hour horizons while also reporting uncertainty calibration and Gaussianity diagnostics.

## What To Show In The Demo

1. Show the input files:

```bash
ls data/raw/train/
```

2. Run the complete pipeline:

```bash
python src/pipeline.py --competition-data data/raw/train/ --output outputs/
```

3. Show satellite classification:

```bash
head outputs/classification_log.csv
```

4. Show prediction format:

```bash
head outputs/submission.csv
```

5. Run evaluation:

```bash
python src/evaluate.py --predictions outputs/submission.csv --actual-data data/raw/train/2026_008.csv --output outputs/evaluation_report.csv
```

6. Show final metrics:

```bash
head outputs/evaluation_report.csv
```

## Honest Prototype Limitations

This is a prototype, not a final optimized model.

Known issues:

- Reset detection is currently sensitive for very low-noise satellites.
- Some satellites are routed to Student-t models even when they may be mostly smooth.
- A few outlier satellites dominate RMSE.
- Some fallback paths are triggered by numerical GP instability.
- Gaussianity diagnostics need refinement to focus on forecast errors rather than raw prediction distribution.

## Next Improvement Phase

After the prototype demonstration, the next work should focus on:

- Tuning reset detection thresholds by constellation.
- Improving classification between clean and irregular satellites.
- Adding targeted handling for worst outlier satellites.
- Improving GP numerical stability.
- Refining uncertainty calibration.
- Reporting constellation-wise and horizon-wise leaderboard metrics.

## Final Positioning

The key message:

> The prototype validates the full workflow: multi-constellation GNSS input, satellite-wise preprocessing, adaptive AI/ML model selection, probabilistic Day 8 prediction, and quantitative evaluation. The foundation is working; the next stage is targeted model refinement.
