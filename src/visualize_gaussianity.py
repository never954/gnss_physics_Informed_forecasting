"""
visualize_gaussianity.py — Visual diagnostics for forecast-error normality.

This script compares Day 8 predictions with actual Day 8 values, then saves
figures that are useful for demonstrating whether the prediction error
distribution is close to normal.

Usage:
    python src/visualize_gaussianity.py \
        --predictions outputs/submission.csv \
        --actual-data data/raw/train/2026_008.csv \
        --output-dir outputs/gaussianity_figures
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import EVAL_HORIZONS_MIN
from src.evaluate import (
    _load_actuals,
    _load_predictions,
    _make_day8_pattern,
    _merge_predictions_actuals,
)
from src.gnss_preprocess import prepare_pipeline_input

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("visualize_gaussianity")


CONSTELLATION_PREFIX: Dict[str, str] = {
    "G": "GPS",
    "E": "Galileo",
    "C": "BeiDou",
    "R": "GLONASS",
    "J": "QZSS",
    "I": "NavIC",
    "S": "SBAS",
}


def build_error_table(
    predictions_csv: str | Path,
    actual_data_path: str | Path,
    cache_dir: str | Path | None = None,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """Load predictions and actual Day 8 values, then return aligned errors."""
    predictions_csv = Path(predictions_csv)
    actual_data_path = Path(actual_data_path)

    preds = _load_predictions(predictions_csv)

    if cache_dir is None:
        cache_dir = predictions_csv.parent / "day8_actuals_cache"
    cache_dir = Path(cache_dir)

    data_dir = actual_data_path.parent if actual_data_path.is_file() else actual_data_path
    actuals_dir = prepare_pipeline_input(
        data_dir=data_dir,
        output_dir=cache_dir,
        n_train_days=8,
        csv_pattern=_make_day8_pattern(actual_data_path),
        force_recompute=force_recompute,
    )
    actuals = _load_actuals(actuals_dir)

    merged = _merge_predictions_actuals(preds, actuals)
    if merged.empty:
        raise ValueError("No overlapping prediction/actual rows found.")

    merged["constellation"] = (
        merged["satellite_id"].astype(str).str[0].map(CONSTELLATION_PREFIX).fillna("Unknown")
    )
    merged["abs_error_ns"] = merged["error_ns"].abs()
    merged["z_score"] = merged["z_score"].replace([np.inf, -np.inf], np.nan)
    merged = merged.dropna(subset=["error_ns", "z_score", "std_ns"])
    return merged


def make_visualizations(errors: pd.DataFrame, output_dir: str | Path) -> None:
    """Generate and save all Gaussianity visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.dpi"] = 140

    errors.to_csv(output_dir / "aligned_prediction_errors.csv", index=False)

    _plot_standardized_hist(errors, output_dir)
    _plot_raw_error_hist(errors, output_dir)
    _plot_qq(errors, output_dir)
    _plot_horizon_boxplot(errors, output_dir)
    _plot_calibration_by_horizon(errors, output_dir)
    _plot_constellation_zscores(errors, output_dir)
    _plot_worst_satellites(errors, output_dir)
    _write_summary(errors, output_dir)

    logger.info(f"Gaussianity figures saved to {output_dir}")


def _normal_pdf_grid(values: np.ndarray, n: int = 300) -> tuple[np.ndarray, np.ndarray]:
    mu = float(np.mean(values))
    sigma = float(np.std(values)) or 1.0
    lo, hi = np.percentile(values, [0.5, 99.5])
    xs = np.linspace(lo, hi, n)
    return xs, stats.norm.pdf(xs, mu, sigma)


def _plot_standardized_hist(errors: pd.DataFrame, output_dir: Path) -> None:
    z = errors["z_score"].dropna().values
    z_plot = z[(z >= -6) & (z <= 6)]

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.histplot(z_plot, bins=70, stat="density", color="#4C78A8", alpha=0.65, ax=ax)
    xs = np.linspace(-6, 6, 500)
    ax.plot(xs, stats.norm.pdf(xs, 0, 1), color="#F58518", lw=3, label="Ideal N(0, 1)")
    for x, label in [(-2, "-2sigma"), (-1, "-1sigma"), (1, "+1sigma"), (2, "+2sigma")]:
        ax.axvline(x, color="#666666", ls="--", lw=1)
        ax.text(x, ax.get_ylim()[1] * 0.92, label, ha="center", va="top", fontsize=10)

    skew = stats.skew(z)
    kurt = stats.kurtosis(z)
    ax.set_title("Standardized Forecast Errors vs Ideal Normal")
    ax.set_xlabel("z = (actual - predicted) / predicted_std")
    ax.set_ylabel("Density")
    ax.legend()
    ax.text(
        0.02,
        0.95,
        f"N = {len(z):,}\nSkew = {skew:.3f}\nExcess kurtosis = {kurt:.3f}",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "01_standardized_error_histogram.png")
    plt.close(fig)


def _plot_raw_error_hist(errors: pd.DataFrame, output_dir: Path) -> None:
    vals = errors["error_ns"].dropna().values
    lo, hi = np.percentile(vals, [1, 99])
    plot_vals = vals[(vals >= lo) & (vals <= hi)]

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.histplot(plot_vals, bins=70, stat="density", color="#54A24B", alpha=0.65, ax=ax)
    xs, pdf = _normal_pdf_grid(plot_vals)
    ax.plot(xs, pdf, color="#E45756", lw=3, label="Normal fit to central 98%")
    ax.axvline(0, color="black", ls="--", lw=1, label="Zero error")
    ax.set_title("Raw Day 8 Forecast Error Distribution")
    ax.set_xlabel("actual - predicted error (ns)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.text(
        0.02,
        0.95,
        f"Central 98% shown\nMAE = {np.mean(np.abs(vals)):.2f} ns\nBias = {np.mean(vals):.2f} ns",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "02_raw_error_histogram.png")
    plt.close(fig)


def _plot_qq(errors: pd.DataFrame, output_dir: Path) -> None:
    z = errors["z_score"].dropna().values
    z = np.clip(z, -8, 8)
    osm, osr = stats.probplot(z, dist="norm", fit=False)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(osm, osr, s=12, alpha=0.45, color="#4C78A8")
    line_min = min(np.min(osm), np.min(osr))
    line_max = max(np.max(osm), np.max(osr))
    ax.plot([line_min, line_max], [line_min, line_max], color="#F58518", lw=3)
    ax.set_title("Q-Q Plot: Standardized Errors vs Normal")
    ax.set_xlabel("Theoretical normal quantiles")
    ax.set_ylabel("Observed standardized-error quantiles")
    ax.text(
        0.05,
        0.95,
        "Closer to diagonal = closer to normal",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "03_qq_plot_standardized_errors.png")
    plt.close(fig)


def _plot_horizon_boxplot(errors: pd.DataFrame, output_dir: Path) -> None:
    horizon_df = errors[errors["horizon_min"].isin(EVAL_HORIZONS_MIN)].copy()
    if horizon_df.empty:
        return
    horizon_df["horizon_label"] = horizon_df["horizon_min"].map(lambda x: f"{int(x)} min")

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.boxplot(
        data=horizon_df,
        x="horizon_label",
        y="abs_error_ns",
        order=[f"{h} min" for h in EVAL_HORIZONS_MIN],
        color="#72B7B2",
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        data=horizon_df,
        x="horizon_label",
        y="abs_error_ns",
        order=[f"{h} min" for h in EVAL_HORIZONS_MIN],
        color="#333333",
        alpha=0.25,
        size=3,
        ax=ax,
    )
    ax.set_title("Absolute Error by Evaluation Horizon")
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("|actual - predicted| (ns)")
    fig.tight_layout()
    fig.savefig(output_dir / "04_abs_error_by_horizon.png")
    plt.close(fig)


def _plot_calibration_by_horizon(errors: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    for horizon in EVAL_HORIZONS_MIN:
        grp = errors[errors["horizon_min"] == horizon]
        if grp.empty:
            continue
        z_abs = grp["z_score"].abs()
        rows.append({
            "horizon_min": horizon,
            "coverage_1sigma": 100 * float((z_abs <= 1).mean()),
            "coverage_2sigma": 100 * float((z_abs <= 2).mean()),
        })
    cal = pd.DataFrame(rows)
    if cal.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(cal["horizon_min"], cal["coverage_1sigma"], marker="o", lw=3, label="Observed +/-1sigma")
    ax.plot(cal["horizon_min"], cal["coverage_2sigma"], marker="o", lw=3, label="Observed +/-2sigma")
    ax.axhline(68.27, color="#4C78A8", ls="--", lw=1.5, label="Ideal +/-1sigma = 68.3%")
    ax.axhline(95.45, color="#F58518", ls="--", lw=1.5, label="Ideal +/-2sigma = 95.5%")
    ax.set_xscale("log")
    ax.set_xticks(EVAL_HORIZONS_MIN)
    ax.set_xticklabels([str(h) for h in EVAL_HORIZONS_MIN])
    ax.set_ylim(0, 105)
    ax.set_title("Uncertainty Calibration by Forecast Horizon")
    ax.set_xlabel("Forecast horizon (minutes, log scale)")
    ax.set_ylabel("Coverage (%)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "05_calibration_by_horizon.png")
    plt.close(fig)


def _plot_constellation_zscores(errors: pd.DataFrame, output_dir: Path) -> None:
    const_order = ["GPS", "Galileo", "BeiDou", "GLONASS", "QZSS", "NavIC", "SBAS"]
    present = [c for c in const_order if c in set(errors["constellation"])]
    if not present:
        return

    g = sns.FacetGrid(
        errors[errors["constellation"].isin(present)].copy(),
        col="constellation",
        col_wrap=3,
        col_order=present,
        sharex=True,
        sharey=False,
        height=3.4,
    )
    g.map_dataframe(
        sns.histplot,
        x="z_score",
        bins=np.linspace(-5, 5, 51),
        stat="density",
        color="#4C78A8",
        alpha=0.65,
    )
    xs = np.linspace(-5, 5, 300)
    for ax in g.axes.flat:
        ax.plot(xs, stats.norm.pdf(xs, 0, 1), color="#F58518", lw=2)
        ax.axvline(0, color="black", ls="--", lw=1)
        ax.set_xlim(-5, 5)
    g.set_axis_labels("standardized error", "density")
    g.fig.suptitle("Standardized Error Distributions by Constellation", y=1.03)
    g.fig.tight_layout()
    g.fig.savefig(output_dir / "06_constellation_standardized_errors.png")
    plt.close(g.fig)


def _plot_worst_satellites(errors: pd.DataFrame, output_dir: Path) -> None:
    sat_stats = (
        errors.groupby("satellite_id")
        .agg(
            n=("error_ns", "size"),
            mae_ns=("abs_error_ns", "mean"),
            bias_ns=("error_ns", "mean"),
            z_skew=("z_score", lambda s: stats.skew(s.dropna()) if len(s.dropna()) >= 4 else np.nan),
            z_kurt=("z_score", lambda s: stats.kurtosis(s.dropna()) if len(s.dropna()) >= 4 else np.nan),
        )
        .reset_index()
    )
    sat_stats["normality_gap"] = sat_stats["z_skew"].abs() + sat_stats["z_kurt"].abs()
    sat_stats = sat_stats.sort_values("mae_ns", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(data=sat_stats, y="satellite_id", x="mae_ns", color="#E45756", ax=ax)
    ax.set_title("Worst Satellites by Mean Absolute Error")
    ax.set_xlabel("MAE (ns)")
    ax.set_ylabel("Satellite")
    fig.tight_layout()
    fig.savefig(output_dir / "07_worst_satellites_by_mae.png")
    plt.close(fig)


def _write_summary(errors: pd.DataFrame, output_dir: Path) -> None:
    z = errors["z_score"].dropna()
    raw = errors["error_ns"].dropna()
    summary = pd.DataFrame([{
        "n": len(errors),
        "n_satellites": errors["satellite_id"].nunique(),
        "mae_ns": float(raw.abs().mean()),
        "rmse_ns": float(np.sqrt(np.mean(raw ** 2))),
        "bias_ns": float(raw.mean()),
        "z_mean": float(z.mean()),
        "z_std": float(z.std()),
        "z_skew": float(stats.skew(z)),
        "z_excess_kurtosis": float(stats.kurtosis(z)),
        "coverage_1sigma_pct": 100 * float((z.abs() <= 1).mean()),
        "coverage_2sigma_pct": 100 * float((z.abs() <= 2).mean()),
    }])
    summary.to_csv(output_dir / "gaussianity_visual_summary.csv", index=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize GNSS forecast-error normality.")
    parser.add_argument("--predictions", default="outputs/submission.csv")
    parser.add_argument("--actual-data", default="data/raw/train/2026_008.csv")
    parser.add_argument("--output-dir", default="outputs/gaussianity_figures")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--force-recompute", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    error_table = build_error_table(
        predictions_csv=args.predictions,
        actual_data_path=args.actual_data,
        cache_dir=args.cache_dir,
        force_recompute=args.force_recompute,
    )
    make_visualizations(error_table, args.output_dir)
