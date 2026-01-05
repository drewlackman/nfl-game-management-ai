"""Train a win-probability model."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

LOG = logging.getLogger(__name__)


def load_processed(processed_dir: Path | str = Path("data/processed")) -> tuple[pd.DataFrame, pd.Series]:
    """Load processed train and validation sets and concatenate for training."""
    processed = Path(processed_dir)
    train_path = processed / "train.parquet"
    val_path = processed / "val.parquet"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("Processed train/val parquet files not found; run build_dataset first.")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    combined = pd.concat([train_df, val_df], ignore_index=True)
    if "home_win" not in combined.columns:
        raise KeyError("Expected 'home_win' target column in processed data.")

    y = combined["home_win"]
    X = combined.drop(columns=["home_win", "game_id"])
    LOG.info("Loaded processed data: %s rows", len(combined))
    return X, y


def build_pipeline(
    categorical: list[str],
    numeric: list[str],
    calibrate: bool = False,
    model_type: str = "gboost",
) -> Pipeline:
    """Create preprocessing + model pipeline."""
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, categorical),
            ("numeric", numeric_transformer, numeric),
        ]
    )

    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise SystemExit("Install xgboost or use --model-type gboost") from exc

        base_model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
        )
    else:
        base_model = GradientBoostingClassifier(random_state=42)
    if calibrate:
        try:
            model = CalibratedClassifierCV(estimator=base_model, method="isotonic", cv=3)
        except TypeError:
            # Backward compatibility for older scikit-learn that uses base_estimator
            model = CalibratedClassifierCV(base_estimator=base_model, method="isotonic", cv=3)
    else:
        model = base_model

    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
    return clf


def train_model(
    processed_dir: Path | str,
    model_dir: Path | str = Path("models"),
    calibrate: bool = False,
    model_type: str = "gboost",
) -> Path:
    """Train the model and save artifact."""
    X, y = load_processed(processed_dir)

    categorical = ["posteam", "defteam", "home_team", "away_team"]
    numeric = [
        "yardline_100",
        "down",
        "ydstogo",
        "log_ydstogo",
        "quarter_seconds_remaining",
        "half_seconds_remaining",
        "game_seconds_remaining",
        "score_differential",
        "posteam_timeouts_remaining",
        "defteam_timeouts_remaining",
        "timeouts_diff",
        "is_home_offense",
        "is_home_defense",
        "goal_to_go",
        "two_min_drill",
    ]

    clf = build_pipeline(categorical, numeric, calibrate=calibrate, model_type=model_type)
    LOG.info("Starting training on %s rows | calibrate=%s | model_type=%s", len(X), calibrate, model_type)
    clf.fit(X, y)

    # Quick evaluation with cross-validation to sanity check.
    if not calibrate:  # skip nested CV when calibrating
        try:
            cv_auc = cross_val_score(clf, X, y, cv=3, scoring="roc_auc", n_jobs=-1)
            LOG.info("CV ROC-AUC: %.3f ± %.3f", cv_auc.mean(), cv_auc.std())
        except Exception as exc:  # pragma: no cover - defensive logging
            LOG.warning("Cross-val failed: %s", exc)

    # In-sample metrics (for rough monitoring; prefer holdout in practice).
    pred_proba = clf.predict_proba(X)[:, 1]
    pred_class = (pred_proba >= 0.5).astype(int)
    try:
        auc = roc_auc_score(y, pred_proba)
        loss = log_loss(y, pred_proba)
        acc = accuracy_score(y, pred_class)
        LOG.info("Train ROC-AUC: %.3f | LogLoss: %.3f | Acc: %.3f", auc, loss, acc)
    except Exception as exc:  # pragma: no cover
        LOG.warning("Metric computation failed: %s", exc)

    out_dir = Path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "wp_model.joblib"
    joblib.dump(clf, model_path)
    LOG.info("Saved model -> %s", model_path)
    return model_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train win probability model.")
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Directory containing processed train/val parquet files.",
    )
    parser.add_argument(
        "--model-dir",
        default="models",
        help="Output directory for serialized model.",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Apply isotonic calibration on top of the base model.",
    )
    parser.add_argument(
        "--model-type",
        choices=["gboost", "xgboost"],
        default="gboost",
        help="Base classifier to use.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(message)s")
    train_model(
        processed_dir=args.processed_dir,
        model_dir=args.model_dir,
        calibrate=args.calibrate,
        model_type=args.model_type,
    )


if __name__ == "__main__":
    main()
