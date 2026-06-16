# GNSS SISE Prediction Pipeline

A physics-informed, modular ML pipeline for predicting Signal-in-Space Errors (SISE) of GNSS navigation satellites (GPS, Galileo, BeiDou, GLONASS) using Gaussian Processes, Bootstrap Monte Carlo, and Student-t Processes.

---

## Prototype Demo Strategy

### Objective
Demonstrate an end-to-end AI/ML prototype that predicts time-varying GNSS satellite clock and ephemeris error buildup from seven days of historical data and forecasts the eighth day at 15-minute intervals.

The prototype addresses the challenge goal:
> **Develop AI/ML based models to predict time-varying patterns of the error buildup between uploaded and modelled values of both satellite clock and ephemeris parameters of navigation satellites.**

### Problem Context
GNSS positioning accuracy is limited by satellite clock bias and ephemeris/orbit prediction errors. These errors evolve after broadcast upload and can degrade positioning and timing reliability.

The challenge provides seven days of satellite error data for multiple GNSS constellations and requires prediction for an unseen eighth day. Evaluation focuses on forecast accuracy at:
- 15 minutes
- 30 minutes
- 1 hour
- 2 hours
- 24 hours

The error distribution is also evaluated for closeness to a normal distribution (Gaussianity).

### Prototype Summary
This prototype builds a complete forecasting pipeline:
1. Ingest multi-constellation GNSS CSV files.
2. Convert broadcast/merged GNSS data into per-satellite SISE time series.
3. Classify each satellite by constellation, orbit type, and reset behavior.
4. Route each satellite to an appropriate probabilistic model.
5. Predict Day 8 errors at 15-minute intervals.
6. Evaluate forecast accuracy, uncertainty calibration, and Gaussianity diagnostics.

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Data Usage
Place your training data files in `data/raw/train/` (see [DATA_ROUTING.md](file:///Users/vedantghule/Downloads/dino.cpp/gnss_my_version/DATA_ROUTING.md) for format details).

* **Training Input**:
  - `data/raw/train/2026_001.csv` to `2026_007.csv` (used for model fitting).
* **Held-out Validation Target**:
  - `data/raw/train/2026_008.csv` (used only for evaluation).

### 3. Demo Command (Run & Evaluate)
Run the complete pipeline and evaluation with:
```bash
python src/pipeline.py --competition-data data/raw/train/ --output outputs/ 2>&1 | tee outputs/pipeline_run.log && python src/evaluate.py --predictions outputs/submission.csv --actual-data data/raw/train/2026_008.csv --output outputs/evaluation_report.csv
```

---

## Pipeline Stages

### 1. Competition Data Preprocessing
The raw competition CSVs contain 78 GNSS columns with broadcast ephemeris, clock terms, constellation metadata, and available IGS truth.
The preprocessing layer converts this into per-satellite 15-minute SISE/error time series:
```
78-column GNSS CSVs -> per-satellite SISE CSVs -> data/processed/sise_intermediate/
```
This lets the downstream ML pipeline work with a consistent time-series format across GPS, Galileo, BeiDou, GLONASS, QZSS, NavIC, and SBAS.

### 2. Satellite Classification
Each satellite is classified by:
- Satellite ID and constellation
- Orbit type: MEO or GEO/GSO
- Reset behavior: clean, regular sawtooth, or irregular
- Recommended model type

The classification output is saved to `outputs/classification_log.csv`.

### 3. Model Routing & The 2×3 Model Matrix
The prototype uses different probabilistic models depending on satellite behavior:

| Satellite Behavior | Model Used | Purpose |
|---|---|---|
| **Clean / smooth** | Gaussian Process | Learns clock drift and orbital/solar periodicity |
| **Regular reset pattern** | Bootstrap Monte Carlo | Generates possible future reset trajectories |
| **Irregular reset pattern** | Student-t Process | Robust to outliers and heavy-tailed jumps |
| **Model failure / unstable case** | Matérn fallback | Ensures prediction coverage |

This is summarized by the model matrix:
```
                    Clean          Regular Sawtooth   Irregular Sawtooth
MEO          →    GP (5-kernel)   Bootstrap MC        Student-t Process
GEO/GSO      →    GP (5-kernel)   Bootstrap MC        Student-t Process
Fallback     →    Matérn GP
```

### 4. Prediction Output Format
Predictions are saved to `outputs/submission.csv`. Each row contains:
`sat_id, timestamp, mean_ns, std_ns, horizon_min`

The model predicts 96 future points per satellite (24 hours × 4 samples/hour).

| sat_id | timestamp | mean_ns | std_ns | horizon_min |
|--------|-----------|---------|--------|-------------|
| G01    | 2024-...  | 1.23    | 0.45   | 15          |
| G01    | 2024-...  | 1.31    | 0.52   | 30          |
| ...    | ...       | ...     | ...    | ...         |

### 5. Evaluation
The evaluation script compares predicted Day 8 values against actual Day 8 processed values.
Full evaluation report is saved to `outputs/evaluation_report.csv`.

Metrics include:
- **MAE / RMSE / MBE**: Mean Absolute Error, Root Mean Squared Error, Mean Bias Error
- **CRPS**: Continuous Ranked Probability Score (probabilistic forecast accuracy)
- **sigma-ratio / coverage**: Uncertainty calibration intervals (1-sigma, 2-sigma)
- **skewness / kurtosis**: Diagnostics for error Gaussianity

---

## Current Prototype Results

Recent evaluation summary:
- **MAE** = 91.179 ns
- **RMSE** = 422.840 ns
- **MBE** = 46.934 ns
- **CRPS** = 63.3724
- **Coverage +/- 1-sigma** = 74.5%
- **Coverage +/- 2-sigma** = 92.9%

*Equivalent average range error:* `91.179 ns x 0.2998 m/ns ~= 27 meters`

#### Horizon-wise Performance:
- **15 min** MAE ~= 53 ns
- **30 min** MAE ~= 55 ns
- **1 hour** MAE ~= 64 ns
- **2 hours** MAE ~= 65 ns
- **24 hours** MAE ~= 135 ns

**Key Insights**:
- Short-horizon forecasts are stronger.
- Error grows toward the 24-hour horizon, as expected.
- The model provides valuable uncertainty estimates (`std_ns`) in addition to point predictions.
- Most satellites are covered across multiple GNSS constellations.
- A small number of difficult satellites dominate the high RMSE.

---

## Project Structure

```
gnss_physics_Informed_forecasting/
├── data/
│   ├── raw/train/          ← Put competition CSV files here
│   ├── raw/sample/         ← Any provided sample/validation data
│   ├── processed/          ← Auto-generated detrended intermediates
│   └── predictions/        ← Per-satellite 96-point CSVs (auto-generated)
├── src/
│   ├── config.py           ← All tunable parameters (edit column names here!)
│   ├── data_loader.py      ← Module 2: Load & validate input
│   ├── sise_compute.py     ← Module 3: Compute SISE from RINEX (if needed)
│   ├── classifier.py       ← Module 4: Orbit + reset pattern classification
│   ├── detrend.py          ← Module 5: Physics detrending
│   ├── reset_detector.py   ← Module 6: Reset & eclipse detection
│   ├── models/
│   │   ├── gp_model.py     ← Module 7: 5-kernel GP (clean satellites)
│   │   ├── bootstrap_mc.py ← Module 8: Bootstrap MC (regular sawtooth)
│   │   ├── student_t.py    ← Module 9: Student-t Process (irregular sawtooth)
│   │   └── matern_fallback.py ← Module 10: Matérn fallback
│   ├── postprocess.py      ← Module 11: Winsorization + formatting
│   └── pipeline.py         ← Module 12: Orchestrator
├── tests/                  ← pytest test suite
├── outputs/
│   └── submission.csv      ← Final competition output
├── CHANGELOG.md
└── DATA_ROUTING.md
```

---

## Demo Guide & Talking Points

### Demo Talking Points
> This prototype demonstrates an end-to-end AI/ML pipeline for predicting GNSS satellite clock and ephemeris error buildup. It ingests seven days of multi-constellation broadcast data, transforms it into satellite-level time series, detects reset and upload behavior, selects a probabilistic forecasting model per satellite, and predicts the eighth day at 15-minute intervals. The system evaluates performance across 15-minute, 30-minute, 1-hour, 2-hour, and 24-hour horizons while also reporting uncertainty calibration and Gaussianity diagnostics.

### What To Show In The Demo
1. **Show the input files**:
   ```bash
   ls data/raw/train/
   ```
2. **Run the complete pipeline**:
   ```bash
   python src/pipeline.py --competition-data data/raw/train/ --output outputs/
   ```
3. **Show satellite classification**:
   ```bash
   head outputs/classification_log.csv
   ```
4. **Show prediction format**:
   ```bash
   head outputs/submission.csv
   ```
5. **Run evaluation**:
   ```bash
   python src/evaluate.py --predictions outputs/submission.csv --actual-data data/raw/train/2026_008.csv --output outputs/evaluation_report.csv
   ```
6. **Show final metrics**:
   ```bash
   head outputs/evaluation_report.csv
   ```

---

## Configuration & Testing

### Configuration
Edit `src/config.py` to:
- Change input column names (`COL_TIMESTAMP`, `COL_CLOCK_ERR`, etc.)
- Tune model hyperparameters
- Add satellite → clock type mappings

### Running Tests
```bash
pytest tests/ -v --cov=src
```

---

## Limitations & Next Steps

### Honest Prototype Limitations
This is a prototype, not a final optimized model. Known issues:
- Reset detection is currently sensitive for very low-noise satellites.
- Some satellites are routed to Student-t models even when they may be mostly smooth.
- A few outlier satellites dominate RMSE.
- Some fallback paths are triggered by numerical GP instability.
- Gaussianity diagnostics need refinement to focus on forecast errors rather than raw prediction distribution.

### Next Improvement Phase
- Tuning reset detection thresholds by constellation.
- Improving classification between clean and irregular satellites.
- Adding targeted handling for worst outlier satellites.
- Improving GP numerical stability.
- Refining uncertainty calibration.
- Reporting constellation-wise and horizon-wise leaderboard metrics.

### Final Positioning
> **The prototype validates the full workflow: multi-constellation GNSS input, satellite-wise preprocessing, adaptive AI/ML model selection, probabilistic Day 8 prediction, and quantitative evaluation. The foundation is working; the next stage is targeted model refinement.**
