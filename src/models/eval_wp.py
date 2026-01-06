"""Evaluate win-probability model on holdout and emit metrics/plot."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

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


def evaluate(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    metrics_dir: Path | str = "outputs/metrics",
    n_bootstrap: int = 0,
    bootstrap_frac: float = 1.0,
) -> None:
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
    lines = [f"ROC-AUC: {auc:.4f}", f"LogLoss: {loss:.4f}", f"Brier: {brier:.4f}"]

    if n_bootstrap > 0:
        rng = np.random.default_rng(42)
        aucs, losses, briers = [], [], []
        n = len(y)
        sample_size = int(n * bootstrap_frac)
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=sample_size)
            y_s = y.iloc[idx]
            p_s = proba[idx]
            aucs.append(roc_auc_score(y_s, p_s))
            losses.append(log_loss(y_s, p_s))
            briers.append(brier_score_loss(y_s, p_s))
        def ci(values):
            return np.percentile(values, [2.5, 97.5])
        auc_ci = ci(aucs)
        loss_ci = ci(losses)
        brier_ci = ci(briers)
        lines.append(f"ROC-AUC 95% CI: [{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]")
        lines.append(f"LogLoss 95% CI: [{loss_ci[0]:.4f}, {loss_ci[1]:.4f}]")
        lines.append(f"Brier 95% CI: [{brier_ci[0]:.4f}, {brier_ci[1]:.4f}]")

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOG.info("Saved metrics -> %s", out_txt)


def main() -> None:
    logging.basicConfig(level="INFO", format="%(levelname)s %(message)s")
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate WP model.")
    parser.add_argument("--metrics-dir", default="outputs/metrics")
    parser.add_argument("--n-bootstrap", type=int, default=0, help="Number of bootstrap resamples for CIs.")
    parser.add_argument(
        "--bootstrap-frac",
        type=float,
        default=1.0,
        help="Fraction of validation set to sample per bootstrap (<=1.0).",
    )
    args = parser.parse_args()
    X, y = load_data()
    model = load_model()
    evaluate(model, X, y, metrics_dir=args.metrics_dir, n_bootstrap=args.n_bootstrap, bootstrap_frac=args.bootstrap_frac)


if __name__ == "__main__":
    main()
