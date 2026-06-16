# GNSS SISE Prediction Pipeline

A physics-informed, modular ML pipeline for predicting Signal-in-Space Errors (SISE) of GNSS navigation satellites (GPS, Galileo, BeiDou, GLONASS) using Gaussian Processes, Bootstrap Monte Carlo, and Student-t Processes.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place your 7-day training data
#    → see DATA_ROUTING.md for exact file format

# 3. Run the pipeline
python src/pipeline.py --data data/raw/train/ --output outputs/

# 4. Predictions appear in outputs/submission.csv
```

## Project Structure

```
Smart_horizon/
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

## The 2×3 Model Matrix

```
                    Clean          Regular Sawtooth   Irregular Sawtooth
MEO          →    GP (5-kernel)   Bootstrap MC        Student-t Process
GEO/GSO      →    GP (5-kernel)   Bootstrap MC        Student-t Process
Fallback     →    Matérn GP
```

## Configuration

Edit `src/config.py` to:
- Change input column names (`COL_TIMESTAMP`, `COL_CLOCK_ERR`, etc.)
- Tune model hyperparameters
- Add satellite → clock type mappings

## Running Tests

```bash
pytest tests/ -v --cov=src
```

## Output Format

`outputs/submission.csv`:

| sat_id | timestamp | mean_ns | std_ns | horizon_min |
|--------|-----------|---------|--------|-------------|
| G01    | 2024-... | 1.23   | 0.45   | 15          |
| G01    | 2024-... | 1.31   | 0.52   | 30          |
| ...    | ...       | ...    | ...    | ...          |

96 rows per satellite × N satellites = full submission.
