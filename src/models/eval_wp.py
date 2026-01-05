"""Evaluate win-probability model on holdout and emit metrics/plot."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve

LOG = logging.getLogger(__name__)


def load_data(processed_dir: Path | str = "data/processed") -> tuple[pd.DataFrame, pd.Series]:
    processed = Path(processed_dir)
    val_path = processed / "val.parquet"
    if not val_path.exists():
        raise FileNotFoundError(f"Validation data not found at {val_path} (run build_dataset).")
    val_df = pd.read_parquet(val_path)
    if "home_win" not in val_df.columns:
        raise KeyError("Expected 'home_win' target column in validation data.")
    y = val_df["home_win"]
    X = val_df.drop(columns=["home_win", "game_id"])
    return X, y


def load_model(model_path: Path | str = "models/wp_model.joblib"):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path} (run train_wp).")
    return joblib.load(path)


def evaluate(model, X: pd.DataFrame, y: pd.Series, metrics_dir: Path | str = "outputs/metrics") -> None:
    metrics_path = Path(metrics_dir)
    metrics_path.mkdir(parents=True, exist_ok=True)

    proba = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, proba)
    loss = log_loss(y, proba)
    brier = brier_score_loss(y, proba)
    LOG.info("Validation — ROC-AUC: %.3f | LogLoss: %.3f | Brier: %.3f", auc, loss, brier)

    # ROC curve
    fpr, tpr, _ = roc_curve(y, proba)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("WP ROC Curve (Validation)")
    plt.legend()
    roc_path = metrics_path / "wp_roc.png"
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150)
    plt.close()

    frac_pos, mean_pred = calibration_curve(y, proba, n_bins=15, strategy="quantile")
    plt.figure(figsize=(5, 5))
    plt.plot(mean_pred, frac_pos, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("WP Calibration (Validation)")
    plt.legend()
    out_plot = metrics_path / "wp_calibration.png"
    plt.tight_layout()
    plt.savefig(out_plot, dpi=150)
    plt.close()
    LOG.info("Saved calibration plot -> %s", out_plot)

    out_txt = metrics_path / "wp_metrics.txt"
    out_txt.write_text(
        f"ROC-AUC: {auc:.4f}\nLogLoss: {loss:.4f}\nBrier: {brier:.4f}\n",
        encoding="utf-8",
    )
    LOG.info("Saved metrics -> %s", out_txt)


def main() -> None:
    logging.basicConfig(level="INFO", format="%(levelname)s %(message)s")
    X, y = load_data()
    model = load_model()
    evaluate(model, X, y)


if __name__ == "__main__":
    main()
