# DATA ROUTING GUIDE

This document tells you exactly **where to put your data**, **what format each file must be in**, and **which script reads from / writes to which location**.

---

## Quick Reference Table

| Stage | Input Path | Output Path | Script |
|-------|-----------|-------------|--------|
| **Competition CSVs** | `data/raw/train/2026_001.csv … 2026_007.csv` | — | You place files here |
| **Module 0: Pre-process** | `data/raw/train/` | `data/processed/sise_intermediate/` | `src/gnss_preprocess.py` |
| **Module 2: Load** | `data/processed/sise_intermediate/` | — | `src/data_loader.py` |
| Detrended residuals | in-memory | `data/processed/<sat>_residual.csv` | `src/detrend.py` |
| Per-satellite predictions | in-memory | `data/predictions/<sat>_prediction.csv` | `src/postprocess.py` |
| Final submission | in-memory | `outputs/submission.csv` | `src/postprocess.py` |

---

## 1. Competition Data Format (Primary)

### Option A — Competition 78-Column CSVs  *(The actual competition format)*

**Place in:** `data/raw/train/`

**Expected filenames:** `2026_001.csv`, `2026_002.csv`, …, `2026_007.csv`
(Day 8 `2026_008.csv` is **not used for training** — Day 8 is the prediction target)

**Run the full pipeline with:**
```bash
python src/pipeline.py --competition-data data/raw/train/ --output outputs/
```

Module 0 (`gnss_preprocess.py`) will automatically:
1. Load all 7 day files (~28,000 rows × 78 cols each)
2. Compute SISE per satellite (see SISE computation below)
3. Resample to 15-min cadence (zero-order hold)
4. Save `data/processed/sise_intermediate/<sat_id>_sise.csv`
5. Hand off to the normal pipeline (Modules 2–12)

---

### Option B — Pre-computed SISE CSV  *(Legacy / testing format)*

**Place in:** `data/raw/train/`
**Expected columns:** `timestamp`, `satellite_id`, `clock_error_ns`, `eph_error_m`, `sise_ns`

**Run with:**
```bash
python src/pipeline.py --data data/raw/train/ --output outputs/
```

---

## 2. SISE Computation (Module 0 Detail)

Module 0 processes each satellite through one of three tiers:

### Tier A — GPS with IGS ground truth (~1.2% of records)

IGS precise products are aligned for these records. SISE is computed as:

```
broadcast_clock(t) = af0 + af1 × (t − t_oc) + af2 × (t − t_oc)²
t_oc               = toe_sec_gps_week  (GPS seconds-of-week)
clock_error_ns     = (broadcast_clock − igs_clock_bias_seconds) × 1e9

broadcast_xyz_m    = keplerian_to_ecef(Keplerian elements, t_gps)
igs_xyz_m          = [igs_x_km, igs_y_km, igs_z_km] × 1000
radial_unit        = igs_xyz_m / |igs_xyz_m|
eph_error_m        = dot(broadcast_xyz_m − igs_xyz_m, radial_unit)

sise_ns            = clock_error_ns + eph_error_m / 0.2998
```

### Tier B — GPS without IGS alignment

```
clock_proxy_ns = (af0 + af1 × (t − t_oc) + af2 × (t − t_oc)²) × 1e9
sise_ns        = clock_proxy_ns
```

### Tier C — Non-GPS (Galileo, GLONASS, BeiDou, QZSS, NavIC, SBAS)

Each constellation uses its own week counter and epoch:
- Galileo → `toe_sec_gal_week` + `gal_week` (epoch: 1999-08-22)
- BeiDou  → `toe_sec_bds_week` + `bds_week` (epoch: 2006-01-01)
- NavIC   → `toe_sec_irn_week` + `irn_week` (epoch: 1999-08-22)
- GLONASS → dt = 0 → `sise_ns = af0 × 1e9`

```
sise_ns = (af0 + af1 × dt + af2 × dt²) × 1e9
```

**Note on non-GPS quality**: Without IGS truth, `sise_ns` represents the absolute broadcast clock bias, not error relative to truth. The temporal pattern (drift, upload resets, harmonics) is accurate for prediction. The absolute level is removed by the detrending step.

### Resampling

Broadcast ephemeris arrives at irregular intervals (~1–2 hours per satellite). After computing per-epoch SISE, Module 0 resamples to a 15-min grid using **zero-order hold** (forward-fill) — matching how receivers use the most recently uploaded ephemeris. Fill is capped at **4 hours** (16 × 15-min steps).

---

## 3. What the Pipeline Reads and Writes

### `src/gnss_preprocess.py` — NEW (Module 0)
- **Reads:** `data/raw/train/2026_NNN.csv` (competition 78-column format)
- **Writes:** `data/processed/sise_intermediate/<sat_id>_sise.csv`
- **Invoked by:** `pipeline.py --competition-data` flag

### `src/data_loader.py`
- **Reads:** `data/processed/sise_intermediate/` (from Module 0) **or** `data/raw/train/` (direct)
- **Writes:** Nothing (returns in-memory DataFrames)

### `src/detrend.py`
- **Reads:** In-memory DataFrames from data_loader
- **Writes:** `data/processed/<sat_id>_residual.csv`

### `src/reset_detector.py`
- **Reads:** Detrended residual (in-memory)
- **Writes:** Nothing (returns ResetEvent list in-memory)

### `src/models/` (GP, BootstrapMC, StudentT, Matern)
- **Reads:** Processed residual + timestamps (in-memory)
- **Writes:** Nothing (returns predictions in-memory)

### `src/postprocess.py`
- **Reads:** Model predictions (in-memory)
- **Writes:** `data/predictions/<sat_id>_prediction.csv` + `outputs/submission.csv`

---

## 4. CLI Reference

### Competition mode (main usage)
```bash
# Full pipeline: 7 training days → Day 8 predictions
python src/pipeline.py \
  --competition-data data/raw/train/ \
  --output           outputs/

# Only specific satellites (useful for testing)
python src/pipeline.py \
  --competition-data data/raw/train/ \
  --sat G01,E05,R03

# Classify only (skip model training)
python src/pipeline.py \
  --competition-data data/raw/train/ \
  --dry-run

# Module 0 standalone (pre-process only)
python src/gnss_preprocess.py \
  --data   data/raw/train/ \
  --output data/processed/sise_intermediate/
```

### Legacy mode (pre-computed SISE)
```bash
python src/pipeline.py --data data/raw/train/ --output outputs/
```

---

## 5. Output File Formats

### Intermediate SISE CSV: `data/processed/sise_intermediate/<sat_id>_sise.csv`
```
timestamp,satellite_id,clock_error_ns,eph_error_m,sise_ns
2026-01-01T00:00:00+00:00,G01,2.15,31.4,2.25
2026-01-01T00:15:00+00:00,G01,2.18,31.4,2.28
...  (672 rows per satellite, 96 × 7 days)
```

### Final Submission: `outputs/submission.csv`
```
sat_id,timestamp,mean_ns,std_ns,horizon_min
G01,2026-01-08T00:00:00Z,1.23,0.45,15
G01,2026-01-08T00:15:00Z,1.31,0.52,30
...  (96 rows × number of satellites)
```

### Classification Log: `outputs/classification_log.csv`
```
sat_id,orbit_type,reset_pattern,model_type,n_resets,mean_reset_interval_hr
G01,MEO,clean,GP,0,0.0
G21,MEO,regular,BootstrapMC,5,31.2
R03,MEO,clean,GP,0,0.0
```

---

## 6. Troubleshooting

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| `No files matching '2026_*.csv'` | Wrong directory or filename | Ensure files are named `2026_NNN.csv` in `--competition-data` dir |
| `KeyError: 'toe_sec_gps_week'` | Column renamed by organiser | Edit `COMP_COL_TOC_GPS` in `src/config.py` Section 11 |
| GLONASS SISE values very large | Expected — af0 is absolute bias | Detrend step removes mean; pattern is still learnable |
| GP training slow | Large dataset on CPU | Use `--gpu` flag or reduce `GP_TRAINING_ITERATIONS` in config |
| `NotPSDError` during GP | Degenerate covariance | Auto-falls back to Matern; check `outputs/classification_log.csv` |
| High kurtosis for non-GPS | Upload pattern not detected as resets | Lower `MAD_THRESHOLD_SIGMA` in config (currently 4.5) |
