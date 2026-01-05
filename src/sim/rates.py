"""Data-driven rate models for go/FG/punt decisions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LOG = logging.getLogger(__name__)


def _split(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=None)
    return X_train, X_val, y_train, y_val


def fit_conversion_model(pbp: pd.DataFrame):
    """Fit a fourth-down conversion classifier."""
    required = {"down", "ydstogo", "yardline_100", "game_seconds_remaining", "score_differential", "yards_gained"}
    missing = required - set(pbp.columns)
    if missing:
        raise KeyError(f"Missing columns for conversion model: {missing}")

    plays = pbp[
        (pbp["down"] == 4)
        & (pbp["play_type"].isin(["run", "pass", "qb_kneel", "qb_scramble"]))
        & pbp["ydstogo"].notna()
        & pbp["yardline_100"].notna()
        & pbp["yards_gained"].notna()
    ].copy()
    if plays.empty:
        raise ValueError("No fourth-down plays available to train conversion model.")

    plays["converted"] = (plays["yards_gained"] >= plays["ydstogo"]).astype(int)
    features = plays[
        ["ydstogo", "yardline_100", "game_seconds_remaining", "score_differential"]
    ].copy()

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("gb", GradientBoostingClassifier(random_state=42)),
        ]
    )

    X_train, X_val, y_train, y_val = _split(pd.concat([features, plays["converted"]], axis=1), "converted")
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    acc = accuracy_score(y_val, preds >= 0.5)
    LOG.info("Conversion model AUC: %.3f | Acc: %.3f | rows: %s", auc, acc, len(plays))
    return model


def fit_fg_model(pbp: pd.DataFrame):
    """Fit a field-goal make probability classifier."""
    required = {"play_type", "yardline_100"}
    missing = required - set(pbp.columns)
    if missing:
        raise KeyError(f"Missing columns for FG model: {missing}")

    kicks = pbp[(pbp["play_type"] == "field_goal") & pbp["yardline_100"].notna()].copy()
    if kicks.empty:
        raise ValueError("No field-goal attempts available to train FG model.")

    distance = kicks["kick_distance"] if "kick_distance" in kicks.columns else kicks["yardline_100"] + 17
    target_col = "fg_good"
    if "field_goal_result" in kicks.columns:
        kicks[target_col] = kicks["field_goal_result"].str.lower().eq("made").astype(int)
    else:
        raise KeyError("Expected field_goal_result column for FG outcomes.")

    kicks[target_col] = kicks[target_col].fillna(0).astype(int)
    features = pd.DataFrame({"kick_distance": distance})

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("gb", GradientBoostingClassifier(random_state=42)),
        ]
    )

    X_train, X_val, y_train, y_val = _split(pd.concat([features, kicks[target_col]], axis=1), target_col)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    acc = accuracy_score(y_val, preds >= 0.5)
    LOG.info("FG model AUC: %.3f | Acc: %.3f | rows: %s", auc, acc, len(kicks))
    return model


def fit_punt_model(pbp: pd.DataFrame):
    """Fit a punt net-yardage regressor."""
    required = {"play_type", "yardline_100"}
    missing = required - set(pbp.columns)
    if missing:
        raise KeyError(f"Missing columns for punt model: {missing}")

    punts = pbp[(pbp["play_type"] == "punt") & pbp["yardline_100"].notna()].copy()
    if punts.empty:
        raise ValueError("No punt plays available to train punt model.")

    if "net" in punts.columns:
        punts["net_yards"] = punts["net"]
    else:
        # Approximate net: use kick_distance minus return_yards when available
        punts["net_yards"] = punts.get("kick_distance", 40).fillna(40) - punts.get("return_yards", 0).fillna(0)

    features = punts[["yardline_100"]].copy()

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("gbr", GradientBoostingRegressor(random_state=42)),
        ]
    )

    X_train, X_val, y_train, y_val = _split(pd.concat([features, punts["net_yards"]], axis=1), "net_yards")
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    LOG.info("Punt model MAE: %.2f | rows: %s", mae, len(punts))
    return model


def save_model(model, path: Path | str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    LOG.info("Saved model -> %s", out)
