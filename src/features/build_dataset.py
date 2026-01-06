"""Build processed dataset for win probability model."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

LOG = logging.getLogger(__name__)


RAW_PATTERN = "pbp_*.parquet"


def load_raw_pbp(raw_dir: Path | str) -> pd.DataFrame:
    """Load concatenated play-by-play parquet files from disk."""
    raw_path = Path(raw_dir)
    files = sorted(raw_path.glob(RAW_PATTERN))
    if not files:
        raise FileNotFoundError(
            f"No parquet files matching {RAW_PATTERN} in {raw_path.resolve()}"
        )
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    LOG.info("Loaded %s rows from %s files", len(df), len(files))
    return df


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Select and derive features plus target."""
    required = [
        "game_id",
        "posteam",
        "defteam",
        "home_team",
        "away_team",
        "yardline_100",
        "down",
        "ydstogo",
        "quarter_seconds_remaining",
        "half_seconds_remaining",
        "game_seconds_remaining",
        "score_differential",
        "posteam_timeouts_remaining",
        "defteam_timeouts_remaining",
        "result",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    working = df[required].copy()
    working.dropna(
        subset=[
            "yardline_100",
            "down",
            "ydstogo",
            "game_seconds_remaining",
            "score_differential",
            "result",
        ],
        inplace=True,
    )

    # Derived indicators
    working["is_home_offense"] = (working["posteam"] == working["home_team"]).astype(
        int
    )
    working["is_home_defense"] = (working["defteam"] == working["home_team"]).astype(
        int
    )
    working["goal_to_go"] = (
        (working["yardline_100"] <= 10) & (working["ydstogo"] <= working["yardline_100"])
    ).astype(int)
    working["two_min_drill"] = (working["game_seconds_remaining"] <= 120).astype(int)
    working["timeouts_diff"] = working["posteam_timeouts_remaining"] - working["defteam_timeouts_remaining"]
    working["log_ydstogo"] = np.log1p(working["ydstogo"])
    working["two_possession"] = (working["score_differential"].abs() >= 9).astype(int)

    # Optional contextual features (defaults when missing)
    working["season"] = df.get("season", pd.Series([0] * len(working))).fillna(0).astype(int)
    roof_series = df.get("roof", pd.Series([None] * len(working)))
    working["roof_closed_or_dome"] = roof_series.str.lower().isin(["closed", "dome", "indoors"]).fillna(0).astype(int)
    surface_series = df.get("surface", pd.Series([None] * len(working)))
    working["surface_synthetic"] = (
        surface_series.str.contains("art", case=False, na=False)
        | surface_series.str.contains("turf", case=False, na=False)
    ).astype(int)
    working["is_neutral_site"] = df.get("neutral_site", pd.Series([0] * len(working))).fillna(0).astype(int)
    working["temp_f"] = df.get("temp", df.get("temperature", pd.Series([np.nan] * len(working)))).astype(float)
    working["wind_mph"] = df.get("wind", pd.Series([np.nan] * len(working))).astype(float)

    features = working.drop(columns=["result"])
    target = (working["result"] > 0).astype(int)
    LOG.info("Engineered features: %s rows, %s columns", len(features), features.shape[1])
    return features, target


def build_dataset(
    raw_dir: Path | str = Path("data/raw"),
    processed_dir: Path | str = Path("data/processed"),
    val_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """Build train/validation parquet files for the WP model."""
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    df = load_raw_pbp(raw_dir)
    X, y = engineer_features(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=random_state, stratify=y
    )

    train = X_train.copy()
    train["home_win"] = y_train
    val = X_val.copy()
    val["home_win"] = y_val

    train_path = processed_path / "train.parquet"
    val_path = processed_path / "val.parquet"
    train.to_parquet(train_path, index=False)
    val.to_parquet(val_path, index=False)

    LOG.info("Wrote train -> %s (%s rows)", train_path, len(train))
    LOG.info("Wrote val   -> %s (%s rows)", val_path, len(val))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build processed dataset for WP model.")
    parser.add_argument("--raw-dir", default="data/raw", help="Directory of raw PBP parquet files.")
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Output directory for processed train/val parquet files.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Validation set proportion.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for splits.",
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
    build_dataset(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        val_size=args.val_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
