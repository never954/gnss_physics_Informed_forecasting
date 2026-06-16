"""
pipeline.py — Module 12: End-to-end pipeline orchestrator.

Routing logic:
    For each satellite:
        0. [Optional] Competition data pre-processing (Module 0)
           → converts 78-column broadcast CSVs to per-satellite SISE CSVs
        1. Load → 2. Classify orbit (GEO/MEO) → 3. Detrend → 4. Detect resets
        → 5. Classify reset pattern → 6. Select model (A/B/C/D)
        → 7. Train model → 8. Predict residual → 9. Add back trend
        → 10. Winsorize → 11. Format & save

CLI usage:
    # Pre-computed SISE data (synthetic / previous format)
    python src/pipeline.py --data data/raw/train/ --output outputs/

    # Competition 78-column CSVs (2026_001.csv … 2026_007.csv)
    python src/pipeline.py --competition-data data/raw/train/ --output outputs/

    # Single-satellite dry run on competition data
    python src/pipeline.py --competition-data data/raw/train/ --sat G01,E05 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Ensure src is on path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    DATA_RAW_TRAIN_DIR,
    DATA_PROCESSED_DIR,
    DATA_PREDICTIONS_DIR,
    OUTPUT_SUBMISSION,
    LOG_LEVEL,
    N_PREDICTION_POINTS,
    PREDICTION_INTERVAL_MIN,
    MIN_STD_NS,
    PREDICTION_STD_SCALE,
    HORIZON_STD_GROWTH_PER_STEP,
    CONSTELLATION_MAD_THRESHOLDS,
    BEIDOU_CONSTELLATIONS,
    BEIDOU_MIN_STD_NS,
)
from src.data_loader import load_satellite_data, get_train_test_split, describe_dataset
from src.classifier import (
    classify_satellite,
    classify_all,
    classification_report,
    OrbitType,
    ResetPattern,
    ModelType,
    SatelliteClassification,
)
from src.detrend import detrend, detrend_all, DetrendResult
from src.reset_detector import detect_resets, mask_eclipses
from src.postprocess import (
    winsorize_predictions,
    format_satellite_output,
    combine_and_save,
    evaluate_gaussianity,
    print_gaussianity_report,
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    data_dir: str | Path = DATA_RAW_TRAIN_DIR,
    output_dir: str | Path = "outputs",
    processed_dir: str | Path = DATA_PROCESSED_DIR,
    predictions_dir: str | Path = DATA_PREDICTIONS_DIR,
    competition_data_dir: Optional[str | Path] = None,
    sat_filter: Optional[List[str]] = None,
    dry_run: bool = False,
    use_gpu: bool = False,
    save_processed: bool = True,
    verbose: bool = False,
) -> Dict:
    """
    Run the full GNSS SISE prediction pipeline.

    Parameters
    ----------
    data_dir : str | Path
        Directory containing training CSV files (pre-computed SISE format).
        Ignored when competition_data_dir is set.
    output_dir : str | Path
        Directory for submission.csv.
    processed_dir : str | Path
        Directory for intermediate detrended residuals.
    predictions_dir : str | Path
        Directory for per-satellite prediction CSVs.
    competition_data_dir : str | Path, optional
        If provided, Module 0 (gnss_preprocess.py) is invoked first to
        convert the competition 78-column broadcast CSVs into per-satellite
        SISE CSVs.  data_dir is then set to the Module 0 output directory
        automatically.  Pass the directory containing 2026_001.csv … 2026_007.csv.
    sat_filter : List[str], optional
        If provided, only process these satellite IDs.
    dry_run : bool
        If True, classify only (skip model training and prediction).
    use_gpu : bool
        Pass to GP models for GPU acceleration.
    save_processed : bool
        Save detrended residuals to disk.
    verbose : bool
        Print per-iteration loss during model training.

    Returns
    -------
    dict with keys: submission, classification_log, gaussianity_metrics
    """
    t_start = time.time()
    logger.info("=" * 60)
    logger.info("GNSS SISE Prediction Pipeline — Starting")
    logger.info("=" * 60)

    # -------------------------------------------------------------------------
    # Step 0 (Optional): Competition data pre-processing
    # -------------------------------------------------------------------------
    if competition_data_dir is not None:
        logger.info("[Step 0] Competition data mode — running Module 0 (gnss_preprocess)...")
        from src.gnss_preprocess import prepare_pipeline_input
        intermediate_dir = Path(processed_dir) / "sise_intermediate"
        data_dir = prepare_pipeline_input(
            data_dir=competition_data_dir,
            output_dir=intermediate_dir,
            n_train_days=7,
        )
        logger.info(f"[Step 0] Module 0 complete. Intermediate CSVs → {data_dir}")

    # -------------------------------------------------------------------------
    # Step 1: Load data
    # -------------------------------------------------------------------------
    logger.info(f"[Step 1] Loading satellite data from {data_dir}...")
    sat_data = load_satellite_data(data_dir)

    if sat_filter:
        sat_data = {k: v for k, v in sat_data.items() if k in sat_filter}
        logger.info(f"Filtered to {len(sat_data)} satellite(s): {sorted(sat_data)}")

    if not sat_data:
        raise ValueError("No satellite data loaded. Check data_dir and CSV format.")

    logger.info(f"Loaded {len(sat_data)} satellites.")
    desc = describe_dataset(sat_data)
    logger.info(f"\n{desc.to_string()}")

    # Build future timestamps per satellite
    _, future_timestamps = get_train_test_split(sat_data)

    # -------------------------------------------------------------------------
    # Step 2: Initial detrend pass (needed for classification)
    # -------------------------------------------------------------------------
    logger.info("[Step 2] Initial detrend pass (for classification)...")
    # We need rough detrended residuals to classify reset patterns.
    # Use GPS defaults for this first pass; re-detrend after classification.
    first_pass_residuals: Dict[str, pd.Series] = {}
    for sat_id, df in sat_data.items():
        from src.config import ORBITAL_PERIODS_HR
        try:
            dr = detrend(
                sat_id=sat_id,
                sise_series=df["sise_ns"],
                constellation="GPS",
                orbital_period_hr=ORBITAL_PERIODS_HR["GPS"],
                save_to_disk=False,
            )
            first_pass_residuals[sat_id] = dr.residual
        except Exception as e:
            logger.warning(f"[{sat_id}] First-pass detrend failed: {e}. Using raw SISE.")
            first_pass_residuals[sat_id] = df["sise_ns"]

    # -------------------------------------------------------------------------
    # Step 3: Detect resets (on rough residuals)
    # -------------------------------------------------------------------------
    logger.info("[Step 3] Detecting resets and eclipses...")
    all_resets: Dict[str, list] = {}
    for sat_id, residual in first_pass_residuals.items():
        # Use constellation-specific MAD threshold (BeiDou needs lower sensitivity
        # to catch smaller-magnitude clock uploads that GPS/GLONASS thresholds miss)
        clf_prelim = classifications_prelim.get(sat_id) if hasattr(locals(), 'classifications_prelim') else None
        constellation = clf_prelim.constellation if clf_prelim else (
            "BEIDOU_MEO" if sat_id.startswith("C") else
            "GLONASS"    if sat_id.startswith("R") else
            "GPS"        if sat_id.startswith("G") else
            "GALILEO"    if sat_id.startswith("E") else
            "UNKNOWN"
        )
        mad_threshold = CONSTELLATION_MAD_THRESHOLDS.get(
            constellation, CONSTELLATION_MAD_THRESHOLDS["UNKNOWN"]
        )
        all_resets[sat_id] = detect_resets(
            residual,
            sat_id=sat_id,
            threshold_sigma=mad_threshold,
        )

    # -------------------------------------------------------------------------
    # Step 4: Classify satellites
    # -------------------------------------------------------------------------
    logger.info("[Step 4] Classifying satellites...")
    classifications = classify_all(sat_data, first_pass_residuals, all_resets)

    clf_report = classification_report(classifications)
    clf_report_path = Path(output_dir) / "classification_log.csv"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    clf_report.to_csv(clf_report_path)
    logger.info(f"Classification log saved to {clf_report_path}")
    logger.info(f"\n{clf_report.to_string()}")

    if dry_run:
        logger.info("[DRY RUN] Stopping after classification. Skipping model training.")
        return {"classification_log": clf_report}

    # -------------------------------------------------------------------------
    # Step 5: Physics detrend with correct constellation periods
    # -------------------------------------------------------------------------
    logger.info("[Step 5] Physics detrending with constellation-specific periods...")
    detrend_results: Dict[str, DetrendResult] = {}
    for sat_id, df in sat_data.items():
        clf = classifications[sat_id]
        try:
            dr = detrend(
                sat_id=sat_id,
                sise_series=df["sise_ns"],
                constellation=clf.constellation,
                orbital_period_hr=clf.orbital_period_hr,
                save_to_disk=save_processed,
                output_dir=str(processed_dir),
            )
            detrend_results[sat_id] = dr
        except Exception as e:
            logger.error(f"[{sat_id}] Detrending failed: {e}. Skipping satellite.")

    # Re-detect resets on properly detrended residuals
    logger.info("[Step 5b] Re-detecting resets on properly detrended residuals...")
    for sat_id, dr in detrend_results.items():
        all_resets[sat_id] = detect_resets(dr.residual, sat_id=sat_id)
        # Update classification with corrected reset info
        clf = classifications[sat_id]
        updated = classify_satellite(
            sat_id=sat_id,
            residual=dr.residual,
            resets=all_resets[sat_id],
        )
        # Preserve orbit/constellation from first classification
        updated.orbit_type = clf.orbit_type
        updated.constellation = clf.constellation
        updated.orbital_period_hr = clf.orbital_period_hr
        classifications[sat_id] = updated

    # -------------------------------------------------------------------------
    # Step 6: Model training and prediction
    # -------------------------------------------------------------------------
    logger.info("[Step 6] Training models and generating predictions...")
    per_satellite_predictions: Dict[str, pd.DataFrame] = {}

    for sat_id in sorted(detrend_results.keys()):
        clf  = classifications[sat_id]
        dr   = detrend_results[sat_id]
        resets = all_resets.get(sat_id, [])
        future_ts = future_timestamps[sat_id]

        logger.info(f"[{sat_id}] → {clf.model_type.value} | {clf.orbit_type.value} | {clf.reset_pattern.value}")

        t_model_start = time.time()
        try:
            # Mask eclipse contamination before model sees the data
            clean_residual = mask_eclipses(dr.residual, resets)

            mean_residual, std_residual = _train_and_predict(
                sat_id=sat_id,
                clf=clf,
                clean_residual=clean_residual,
                timestamps=dr.timestamps,
                resets=resets,
                future_ts=future_ts,
                use_gpu=use_gpu,
                verbose=verbose,
            )

            # Re-add physics trend — anchored to last observed SISE
            # Why anchoring? The polynomial trend is fitted over 7 days and may
            # not exactly match the last observed value. If we extrapolate the raw
            # polynomial, we inherit whatever offset exists at Day 7 end, which
            # compounds over 24 hours of prediction. Anchoring forces the trend
            # at t=0 (first prediction point) to equal the last observed SISE,
            # eliminating the systematic MBE seen in evaluation.
            trend_future = dr.predict_trend(future_ts)

            # Compute trend value at the last training timestamp
            last_ts_idx = pd.DatetimeIndex([dr.timestamps[-1]])
            trend_at_last = dr.predict_trend(last_ts_idx)[0]

            # Last observed SISE (use last valid value)
            last_observed_sise = float(dr.sise_ns.dropna().iloc[-1])

            # Anchoring offset: shift future trend so it starts from observed state
            anchor_offset = last_observed_sise - trend_at_last

            mean_full = mean_residual + trend_future + anchor_offset
            std_full  = std_residual   # Uncertainty not affected by deterministic trend

            logger.debug(
                f"[{sat_id}] Anchor offset: {anchor_offset:.3f} ns "
                f"(last_obs={last_observed_sise:.3f}, trend_at_last={trend_at_last:.3f})"
            )

            # Winsorize (Gaussianity safety net)
            mean_final, std_final = winsorize_predictions(
                mean_pred=mean_full,
                std_pred=std_full,
                train_series=dr.sise_ns,
                sat_id=sat_id,
            )

            # ── Fix 4+7: Std floor based on detrended residual ──────────────
            # All satellites: floor = max(MIN_STD_NS, residual_std × PREDICTION_STD_SCALE)
            # - residual_std captures the detrended within-segment noise (~1-15 ns)
            # - This is correct for GPS/GLONASS/Galileo (smooth, infrequent resets)
            #
            # BeiDou additional minimum (BEIDOU_MIN_STD_NS = 50 ns):
            # - BeiDou errors (6-304 ns) come from upload events changing the
            #   broadcast polynomial between Day 7 and Day 8 — NOT from the
            #   absolute clock bias (~100k-300k ns for GEO satellites).
            # - residual_std is too small (5-15 ns within a clean segment).
            # - A 50 ns floor covers 40/43 BeiDou satellites' error range
            #   (break-even at 11 ns; only C36=6.5 and C38=12 get a tiny CRPS
            #   penalty, all others benefit).
            residual_std  = float(dr.residual.dropna().std())
            sat_std_floor = max(MIN_STD_NS, residual_std * PREDICTION_STD_SCALE)
            if clf.constellation in BEIDOU_CONSTELLATIONS:
                sat_std_floor = max(sat_std_floor, BEIDOU_MIN_STD_NS)
            std_final = np.maximum(std_final, sat_std_floor)
            logger.debug(
                f"[{sat_id}] Std floor: {sat_std_floor:.2f} ns "
                f"(residual_std={residual_std:.2f} ns, "
                f"beidou_min={BEIDOU_MIN_STD_NS if clf.constellation in BEIDOU_CONSTELLATIONS else 'n/a'})"
            )

            # ── Fix 5: Horizon-growing uncertainty ────────────────────────
            # Prediction uncertainty must grow with horizon:
            #   - Clock drift compounds over 24 h
            #   - Probability of a BeiDou upload increases with time
            # At step k: std × (1 + HORIZON_STD_GROWTH_PER_STEP × k)
            # At step 96 (24 h): std × ~2.9× relative to step 1
            horizon_steps  = np.arange(1, len(std_final) + 1)
            horizon_factor = 1.0 + HORIZON_STD_GROWTH_PER_STEP * horizon_steps
            std_final      = std_final * horizon_factor

            last_train_ts = dr.timestamps[-1]
            sat_df = format_satellite_output(
                sat_id=sat_id,
                future_timestamps=future_ts,
                mean_ns=mean_final,
                std_ns=std_final,
                last_train_timestamp=last_train_ts,
            )
            per_satellite_predictions[sat_id] = sat_df

            t_elapsed = time.time() - t_model_start
            logger.info(f"[{sat_id}] Done in {t_elapsed:.1f}s")

        except Exception as e:
            logger.error(f"[{sat_id}] Model failed: {e}. Falling back to Matérn.")
            try:
                mean_residual, std_residual = _fallback_predict(
                    sat_id=sat_id,
                    dr=dr,
                    future_ts=future_ts,
                    use_gpu=use_gpu,
                )
                trend_future = dr.predict_trend(future_ts)
                last_ts_idx = pd.DatetimeIndex([dr.timestamps[-1]])
                trend_at_last = dr.predict_trend(last_ts_idx)[0]
                last_observed_sise = float(dr.sise_ns.dropna().iloc[-1])
                anchor_offset = last_observed_sise - trend_at_last
                mean_final, std_final = winsorize_predictions(
                    mean_pred=mean_residual + trend_future + anchor_offset,
                    std_pred=std_residual,
                    train_series=dr.sise_ns,
                    sat_id=sat_id,
                )
                # Fix 4+7: std floor — fallback path
                residual_std  = float(dr.residual.dropna().std())
                sat_std_floor = max(MIN_STD_NS, residual_std * PREDICTION_STD_SCALE)
                if clf.constellation in BEIDOU_CONSTELLATIONS:
                    sat_std_floor = max(sat_std_floor, BEIDOU_MIN_STD_NS)
                std_final = np.maximum(std_final, sat_std_floor)
                # Fix 5: horizon-growing uncertainty
                horizon_steps  = np.arange(1, len(std_final) + 1)
                std_final      = std_final * (1.0 + HORIZON_STD_GROWTH_PER_STEP * horizon_steps)
                sat_df = format_satellite_output(
                    sat_id=sat_id,
                    future_timestamps=future_ts,
                    mean_ns=mean_final,
                    std_ns=std_final,
                    last_train_timestamp=dr.timestamps[-1],
                )
                per_satellite_predictions[sat_id] = sat_df
                logger.info(f"[{sat_id}] Fallback successful.")
            except Exception as e2:
                logger.error(f"[{sat_id}] Fallback also failed: {e2}. Satellite skipped.")

    # -------------------------------------------------------------------------
    # Step 7: Save outputs
    # -------------------------------------------------------------------------
    logger.info("[Step 7] Saving outputs...")
    submission = combine_and_save(
        per_satellite_dfs=per_satellite_predictions,
        output_dir=output_dir,
        predictions_dir=predictions_dir,
    )

    # -------------------------------------------------------------------------
    # Step 8: Gaussianity evaluation (diagnostic — on predictions themselves)
    # -------------------------------------------------------------------------
    logger.info("[Step 8] Gaussianity evaluation (diagnostic)...")
    gaussianity = evaluate_gaussianity(submission)
    print_gaussianity_report(gaussianity)
    gaussianity_path = Path(output_dir) / "gaussianity_report.csv"
    gaussianity.to_csv(gaussianity_path, index=False)
    logger.info(f"Gaussianity report saved to {gaussianity_path}")

    t_total = time.time() - t_start
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Pipeline complete in {t_total:.1f}s")
    logger.info(f"  Satellites processed: {len(per_satellite_predictions)}/{len(sat_data)}")
    logger.info(f"  Submission: outputs/submission.csv")
    logger.info(f"  Classification log: outputs/classification_log.csv")
    logger.info(f"  Gaussianity report: outputs/gaussianity_report.csv")
    logger.info(f"{'=' * 60}\n")

    return {
        "submission": submission,
        "classification_log": clf_report,
        "gaussianity_metrics": gaussianity,
    }


# ---------------------------------------------------------------------------
# Model dispatch
# ---------------------------------------------------------------------------

def _train_and_predict(
    sat_id: str,
    clf: SatelliteClassification,
    clean_residual: pd.Series,
    timestamps: pd.DatetimeIndex,
    resets: list,
    future_ts: pd.DatetimeIndex,
    use_gpu: bool,
    verbose: bool,
) -> tuple:
    """Route to correct model and return (mean_residual, std_residual)."""

    model_type = clf.model_type

    if model_type == ModelType.GP:
        from src.models.gp_model import GPModel
        model = GPModel(
            sat_id=sat_id,
            orbital_period_hr=clf.orbital_period_hr,
            use_gpu=use_gpu,
        )
        model.fit(timestamps, clean_residual, verbose=verbose)
        return model.predict(future_ts)

    elif model_type == ModelType.BOOTSTRAP:
        from src.models.bootstrap_mc import BootstrapMCModel
        model = BootstrapMCModel(
            sat_id=sat_id,
            orbital_period_hr=clf.orbital_period_hr,
            constellation=clf.constellation,
        )
        model.fit(timestamps, clean_residual, resets)
        return model.predict(future_ts)

    elif model_type == ModelType.STUDENT_T:
        from src.models.student_t import StudentTModel
        model = StudentTModel(
            sat_id=sat_id,
            orbital_period_hr=clf.orbital_period_hr,
            use_gpu=use_gpu,
        )
        model.fit(timestamps, clean_residual, verbose=verbose)
        return model.predict(future_ts)

    else:  # MATERN fallback
        return _fallback_predict(sat_id, None, future_ts, use_gpu, timestamps, clean_residual)


def _fallback_predict(
    sat_id: str,
    dr: Optional[DetrendResult],
    future_ts: pd.DatetimeIndex,
    use_gpu: bool,
    timestamps: Optional[pd.DatetimeIndex] = None,
    residual: Optional[pd.Series] = None,
) -> tuple:
    """Matérn fallback prediction."""
    from src.models.matern_fallback import MaternFallbackModel

    if dr is not None:
        ts = dr.timestamps
        res = dr.residual
    else:
        ts = timestamps
        res = residual

    model = MaternFallbackModel(sat_id=sat_id, use_gpu=use_gpu)
    model.fit(ts, res)
    return model.predict(future_ts)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="GNSS SISE Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Competition 78-column CSVs (main usage)
  python src/pipeline.py --competition-data data/raw/train/ --output outputs/

  # Pre-computed SISE data (synthetic / legacy format)
  python src/pipeline.py --data data/raw/train/ --output outputs/

  # Process only G01 and E05 (competition mode)
  python src/pipeline.py --competition-data data/raw/train/ --sat G01,E05

  # Classify only — no model training
  python src/pipeline.py --competition-data data/raw/train/ --dry-run

  # With GPU acceleration
  python src/pipeline.py --competition-data data/raw/train/ --gpu
        """,
    )
    parser.add_argument("--data",             default=DATA_RAW_TRAIN_DIR,   help="Pre-computed SISE CSV directory")
    parser.add_argument("--competition-data", default=None,                  help="Competition 78-column CSV directory (2026_NNN.csv files)")
    parser.add_argument("--output",           default="outputs",             help="Output directory")
    parser.add_argument("--processed",        default=DATA_PROCESSED_DIR,   help="Processed data directory")
    parser.add_argument("--predictions",      default=DATA_PREDICTIONS_DIR, help="Per-satellite predictions dir")
    parser.add_argument("--sat",              default=None,                  help="Comma-separated satellite IDs")
    parser.add_argument("--dry-run",          action="store_true",           help="Classify only, skip training")
    parser.add_argument("--gpu",              action="store_true",           help="Use GPU (if available)")
    parser.add_argument("--verbose",          action="store_true",           help="Print per-iteration loss")
    parser.add_argument("--no-save-processed",action="store_true",           help="Don't save intermediate residuals")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    sat_filter = [s.strip() for s in args.sat.split(",")] if args.sat else None

    run_pipeline(
        data_dir=args.data,
        competition_data_dir=getattr(args, "competition_data", None),
        output_dir=args.output,
        processed_dir=args.processed,
        predictions_dir=args.predictions,
        sat_filter=sat_filter,
        dry_run=args.dry_run,
        use_gpu=args.gpu,
        save_processed=not args.no_save_processed,
        verbose=args.verbose,
    )
