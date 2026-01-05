"""Evaluate conversion, field-goal, and punt models with metrics and calibration plots."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, roc_auc_score

LOG = logging.getLogger(__name__)
RAW_PATTERN = "pbp_*.parquet"


def load_raw_pbp(raw_dir: Path | str) -> pd.DataFrame:
    raw_path = Path(raw_dir)
    files = sorted(raw_path.glob(RAW_PATTERN))
    if not files:
        raise FileNotFoundError(f"No parquet files matching {RAW_PATTERN} in {raw_path.resolve()}")
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    LOG.info("Loaded %s rows from %s files", len(df), len(files))
    return df


def evaluate_conversion(model, pbp: pd.DataFrame, metrics_dir: Path) -> None:
    plays = pbp[
        (pbp["down"] == 4)
        & (pbp["play_type"].isin(["run", "pass", "qb_kneel", "qb_scramble"]))
        & pbp["ydstogo"].notna()
        & pbp["yardline_100"].notna()
        & pbp["yards_gained"].notna()
    ].copy()
    if plays.empty:
        LOG.warning("No fourth-down plays to evaluate conversion model.")
        return

    plays["converted"] = (plays["yards_gained"] >= plays["ydstogo"]).astype(int)
    X = plays[["ydstogo", "yardline_100", "game_seconds_remaining", "score_differential"]]
    y = plays["converted"]

    prob = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, prob)
    brier = brier_score_loss(y, prob)
    ll = log_loss(y, prob)
    baseline = np.full_like(y, y.mean(), dtype=float)
    lift_auc = auc - 0.5
    lift_brier = brier_score_loss(y, baseline) - brier
    LOG.info("Conversion — AUC: %.3f | LogLoss: %.3f | Brier: %.3f | lift_auc: %.3f | lift_brier: %.3f", auc, ll, brier, lift_auc, lift_brier)

    frac_pos, mean_pred = calibration_curve(y, prob, n_bins=15, strategy="quantile")
    plt.figure(figsize=(5, 5))
    plt.plot(mean_pred, frac_pos, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Conversion Calibration")
    plt.legend()
    out_plot = metrics_dir / "conversion_calibration.png"
    plt.tight_layout()
    plt.savefig(out_plot, dpi=150)
    plt.close()

    # Error histogram
    plt.figure(figsize=(5, 4))
    plt.hist(prob - y, bins=30, alpha=0.7)
    plt.xlabel("Predicted - Actual")
    plt.ylabel("Count")
    plt.title("Conversion Error Histogram")
    err_plot = metrics_dir / "conversion_error_hist.png"
    plt.tight_layout()
    plt.savefig(err_plot, dpi=150)
    plt.close()

    out_txt = metrics_dir / "conversion_metrics.txt"
    out_txt.write_text(
        f"AUC: {auc:.4f}\nLogLoss: {ll:.4f}\nBrier: {brier:.4f}\nLift_AUC_vs_random: {lift_auc:.4f}\nLift_Brier_vs_mean: {lift_brier:.4f}\n",
        encoding="utf-8",
    )


def evaluate_fg(model, pbp: pd.DataFrame, metrics_dir: Path) -> None:
    kicks = pbp[(pbp["play_type"] == "field_goal") & pbp["yardline_100"].notna()].copy()
    if kicks.empty:
        LOG.warning("No field-goal attempts to evaluate FG model.")
        return

    distance = kicks["kick_distance"] if "kick_distance" in kicks.columns else kicks["yardline_100"] + 17
    if "field_goal_result" not in kicks.columns:
        LOG.warning("field_goal_result missing; skipping FG eval.")
        return
    y = kicks["field_goal_result"].str.lower().eq("made").astype(int)
    X = pd.DataFrame({"kick_distance": distance})

    prob = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, prob)
    brier = brier_score_loss(y, prob)
    ll = log_loss(y, prob)
    baseline = np.full_like(y, y.mean(), dtype=float)
    lift_auc = auc - 0.5
    lift_brier = brier_score_loss(y, baseline) - brier
    LOG.info("FG — AUC: %.3f | LogLoss: %.3f | Brier: %.3f | lift_auc: %.3f | lift_brier: %.3f", auc, ll, brier, lift_auc, lift_brier)

    frac_pos, mean_pred = calibration_curve(y, prob, n_bins=15, strategy="quantile")
    plt.figure(figsize=(5, 5))
    plt.plot(mean_pred, frac_pos, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("FG Calibration")
    plt.legend()
    out_plot = metrics_dir / "fg_calibration.png"
    plt.tight_layout()
    plt.savefig(out_plot, dpi=150)
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.hist(prob - y, bins=30, alpha=0.7)
    plt.xlabel("Predicted - Actual")
    plt.ylabel("Count")
    plt.title("FG Error Histogram")
    err_plot = metrics_dir / "fg_error_hist.png"
    plt.tight_layout()
    plt.savefig(err_plot, dpi=150)
    plt.close()

    out_txt = metrics_dir / "fg_metrics.txt"
    out_txt.write_text(
        f"AUC: {auc:.4f}\nLogLoss: {ll:.4f}\nBrier: {brier:.4f}\nLift_AUC_vs_random: {lift_auc:.4f}\nLift_Brier_vs_mean: {lift_brier:.4f}\n",
        encoding="utf-8",
    )


def evaluate_punt(model, pbp: pd.DataFrame, metrics_dir: Path) -> None:
    punts = pbp[(pbp["play_type"] == "punt") & pbp["yardline_100"].notna()].copy()
    if punts.empty:
        LOG.warning("No punts to evaluate punt model.")
        return

    if "net" in punts.columns:
        y = punts["net"]
    else:
        y = punts.get("kick_distance", 40).fillna(40) - punts.get("return_yards", 0).fillna(0)
    X = pd.DataFrame({"yardline_100": punts["yardline_100"]})

    pred = model.predict(X)
    mae = mean_absolute_error(y, pred)
    baseline = np.full_like(y, y.mean(), dtype=float)
    baseline_mae = mean_absolute_error(y, baseline)
    lift_mae = baseline_mae - mae
    LOG.info("Punt — MAE: %.2f | baseline MAE: %.2f | lift: %.2f", mae, baseline_mae, lift_mae)

    plt.figure(figsize=(5, 4))
    plt.hist(pred - y, bins=30, alpha=0.7)
    plt.xlabel("Prediction - Actual Net Yards")
    plt.ylabel("Count")
    plt.title("Punt Error Histogram")
    err_plot = metrics_dir / "punt_error_hist.png"
    plt.tight_layout()
    plt.savefig(err_plot, dpi=150)
    plt.close()

    out_txt = metrics_dir / "punt_metrics.txt"
    out_txt.write_text(
        f"MAE: {mae:.4f}\nBaseline_MAE: {baseline_mae:.4f}\nLift_MAE_vs_mean: {lift_mae:.4f}\n",
        encoding="utf-8",
    )


def main() -> None:
    logging.basicConfig(level="INFO", format="%(levelname)s %(message)s")
    raw_dir = Path("data/raw")
    model_dir = Path("models/rates")
    metrics_dir = Path("outputs/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    pbp = load_raw_pbp(raw_dir)

    conv_path = model_dir / "conversion.joblib"
    fg_path = model_dir / "fg.joblib"
    punt_path = model_dir / "punt.joblib"
    if conv_path.exists():
        evaluate_conversion(joblib.load(conv_path), pbp, metrics_dir)
    else:
        LOG.warning("Conversion model not found at %s", conv_path)
    if fg_path.exists():
        evaluate_fg(joblib.load(fg_path), pbp, metrics_dir)
    else:
        LOG.warning("FG model not found at %s", fg_path)
    if punt_path.exists():
        evaluate_punt(joblib.load(punt_path), pbp, metrics_dir)
    else:
        LOG.warning("Punt model not found at %s", punt_path)


if __name__ == "__main__":
    main()
