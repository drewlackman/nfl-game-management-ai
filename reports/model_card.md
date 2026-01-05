# Model Card — NFL Win Probability + Decision Models

## Overview
Models to support NFL in-game decisions (4th down): a win-probability (WP) classifier plus data-driven rates for conversion, field-goal makes, and punt net yards. Outputs feed simulations and UI/API/agent surfaces.

## Data
- Source: Play-by-play via `nfl_data_py` (nflfastR data), seasons: [fill, e.g., 2021-2023].
- Splits: train/val from `src/features/build_dataset.py` (stratified split, 80/20).
- Processed artifacts: `data/processed/train.parquet`, `data/processed/val.parquet`.

## Features (WP)
- Situation: down, ydstogo (+log), yardline_100, score diff, game/half/qtr seconds remaining.
- Timeouts: offense/defense, diff.
- Context flags: goal_to_go, two_min_drill, is_home_offense/defense.
- Categorical: posteam, defteam, home_team, away_team (one-hot).

## Models
- WP: GradientBoosting or XGBoost (`--model-type xgboost`) with optional isotonic calibration (`--calibrate`).
- Rates: GradientBoosting for conversion/FG (classification), GradientBoostingRegressor for punt net yards.

## Validation
- WP metrics: ROC-AUC, LogLoss, Brier; calibration curve (quantile bins).
- Rate metrics: Conversion/FG AUC/LogLoss/Brier; punt MAE.
- Plots: `outputs/metrics/wp_calibration.png`, `wp_roc.png`, `conversion_calibration.png`, `fg_calibration.png`, `*_error_hist.png`.

## Limitations
- No weather/injury/roster strength; no opponent-specific priors.
- Uses historical averages for conversion/FG/punt unless rate models trained; uncertainty not propagated.
- WP target: home win indicator; does not model spread or overtime nuances.
- No play-clock/penalty risk modeling.

## Intended Use
- Decision support and analysis; not for gambling or productionized live betting.
- Human-in-the-loop: pair outputs with coaching context.

## Next Steps
- Add opponent/venue/weather features; drive/series context.
- Quantify uncertainty (e.g., bootstrap, Bayesian calibration) and propagate through sim.
- Integrate learned conversion/FG/punt models trained on latest seasons; add opponent-adjusted priors.
- Benchmark alternative WP models (LightGBM, CatBoost) and calibration methods.
- Add latency/throughput benchmarks for API/Streamlit; cache models/rate models.

## Repro
- Train: `python -m src.models.train_wp --model-type xgboost --calibrate`
- Rates: `python -m src.sim.train_rates`
- Eval: `python -m src.models.eval_wp`; `python -m src.sim.eval_rates`
