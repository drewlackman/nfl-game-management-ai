"""Train rate models for conversion, field goal, and punt outcomes."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.sim import rates

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


def train_and_save(raw_dir: Path | str = "data/raw", model_dir: Path | str = "models/rates") -> None:
    pbp = load_raw_pbp(raw_dir)

    conv_model = rates.fit_conversion_model(pbp)
    fg_model = rates.fit_fg_model(pbp)
    punt_model = rates.fit_punt_model(pbp)

    rates.save_model(conv_model, Path(model_dir) / "conversion.joblib")
    rates.save_model(fg_model, Path(model_dir) / "fg.joblib")
    rates.save_model(punt_model, Path(model_dir) / "punt.joblib")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train rate models for fourth-down sim.")
    parser.add_argument("--raw-dir", default="data/raw", help="Directory containing pbp_*.parquet files.")
    parser.add_argument("--model-dir", default="models/rates", help="Output directory for rate models.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(message)s")
    train_and_save(raw_dir=args.raw_dir, model_dir=args.model_dir)


if __name__ == "__main__":
    main()
