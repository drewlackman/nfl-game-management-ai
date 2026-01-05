"""Evaluate fourth-down decisions using the win-probability model.

This module is intentionally heuristic: it uses simple probability estimates
for conversion, field goal, and punt outcomes, then feeds resulting states
into the trained WP model.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import time


@dataclass
class GameState:
    """Minimal game state needed for WP evaluation."""

    home_team: str
    away_team: str
    posteam: str
    defteam: str
    yardline_100: float
    down: int
    ydstogo: float
    quarter_seconds_remaining: float
    half_seconds_remaining: float
    game_seconds_remaining: float
    home_score: int
    away_score: int
    posteam_timeouts: int
    defteam_timeouts: int

    def posteam_score(self) -> int:
        return self.home_score if self.posteam == self.home_team else self.away_score

    def defteam_score(self) -> int:
        return self.away_score if self.posteam == self.home_team else self.home_score

    def score_differential(self) -> int:
        return self.posteam_score() - self.defteam_score()

    def is_home_offense(self) -> int:
        return int(self.posteam == self.home_team)

    def is_home_defense(self) -> int:
        return int(self.defteam == self.home_team)

    def to_features(self) -> pd.DataFrame:
        """Build model-ready feature row."""
        goal_to_go = int(self.yardline_100 <= 10 and self.ydstogo <= self.yardline_100)
        two_min_drill = int(self.game_seconds_remaining <= 120)
        timeouts_diff = int(self.posteam_timeouts - self.defteam_timeouts)
        data = {
            "posteam": self.posteam,
            "defteam": self.defteam,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "yardline_100": float(self.yardline_100),
            "down": int(self.down),
            "ydstogo": float(self.ydstogo),
            "log_ydstogo": float(np.log1p(self.ydstogo)),
            "quarter_seconds_remaining": float(self.quarter_seconds_remaining),
            "half_seconds_remaining": float(self.half_seconds_remaining),
            "game_seconds_remaining": float(self.game_seconds_remaining),
            "score_differential": float(self.score_differential()),
            "posteam_timeouts_remaining": int(self.posteam_timeouts),
            "defteam_timeouts_remaining": int(self.defteam_timeouts),
            "is_home_offense": self.is_home_offense(),
            "is_home_defense": self.is_home_defense(),
            "goal_to_go": goal_to_go,
            "two_min_drill": two_min_drill,
            "timeouts_diff": timeouts_diff,
        }
        return pd.DataFrame([data])


def load_model(model_path: Path | str = Path("models/wp_model.joblib")):
    """Load serialized WP model."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Train it first.")
    return joblib.load(path)


@dataclass
class RateModels:
    conversion_model: Optional[object] = None
    fg_model: Optional[object] = None
    punt_model: Optional[object] = None


def load_rate_models(model_dir: Path | str = Path("models/rates")) -> RateModels:
    """Load rate models if present; missing files are ignored."""
    model_dir = Path(model_dir)
    conv = fg = punt = None
    conv_path = model_dir / "conversion.joblib"
    fg_path = model_dir / "fg.joblib"
    punt_path = model_dir / "punt.joblib"

    if conv_path.exists():
        conv = joblib.load(conv_path)
    if fg_path.exists():
        fg = joblib.load(fg_path)
    if punt_path.exists():
        punt = joblib.load(punt_path)

    return RateModels(conversion_model=conv, fg_model=fg, punt_model=punt)


def flip_possession(state: GameState, new_yardline_100: float) -> GameState:
    """Return a new GameState with possession flipped."""
    new_state = replace(
        state,
        posteam=state.defteam,
        defteam=state.posteam,
        yardline_100=new_yardline_100,
        posteam_timeouts=state.defteam_timeouts,
        defteam_timeouts=state.posteam_timeouts,
    )
    return new_state


def estimate_go_conversion_prob(state: GameState, rate_models: RateModels | None = None) -> float:
    """Estimate fourth-down conversion probability."""
    if rate_models and rate_models.conversion_model is not None:
        features = pd.DataFrame(
            {
                "ydstogo": [state.ydstogo],
                "yardline_100": [state.yardline_100],
                "game_seconds_remaining": [state.game_seconds_remaining],
                "score_differential": [state.score_differential()],
            }
        )
        try:
            return float(rate_models.conversion_model.predict_proba(features)[0, 1])
        except Exception:  # pragma: no cover - fallback safety
            pass

    base = 0.68
    yard_penalty = 0.025 * max(state.ydstogo - 1, 0)
    down_penalty = 0.04 * max(state.down - 1, 0)
    field_penalty = 0.002 * max(state.yardline_100 - 50, 0)  # tougher closer to goal
    prob = base - yard_penalty - down_penalty - field_penalty
    return float(np.clip(prob, 0.05, 0.95))


def estimate_fg_prob(state: GameState, rate_models: RateModels | None = None) -> float:
    """Estimate field-goal success probability."""
    distance = state.yardline_100 + 17  # line of scrimmage + 17 yards

    if rate_models and rate_models.fg_model is not None:
        features = pd.DataFrame({"kick_distance": [distance]})
        try:
            return float(rate_models.fg_model.predict_proba(features)[0, 1])
        except Exception:  # pragma: no cover
            pass

    distance = state.yardline_100 + 17  # line of scrimmage + 17 yards
    prob = 0.97 - 0.018 * max(distance - 33, 0) / 5
    return float(np.clip(prob, 0.1, 0.98))


def estimate_punt_net_yards(state: GameState, rate_models: RateModels | None = None) -> float:
    """Estimate net punt yards."""
    if rate_models and rate_models.punt_model is not None:
        features = pd.DataFrame({"yardline_100": [state.yardline_100]})
        try:
            return float(rate_models.punt_model.predict(features)[0])
        except Exception:  # pragma: no cover
            pass

    base = 44.0
    short_field_penalty = max(0, 50 - state.yardline_100) * 0.12
    return float(np.clip(base - short_field_penalty, 20, 55))


def predict_wp(model, state: GameState) -> float:
    """Predict home win probability given a state."""
    proba = model.predict_proba(state.to_features())[0, 1]
    return float(proba)


def simulate_go_for_it(model, state: GameState, rate_models: RateModels | None = None) -> Tuple[float, Dict[str, float]]:
    conv_prob = estimate_go_conversion_prob(state, rate_models)
    yards_gained = max(state.ydstogo, 1)

    # Success state
    new_yard = max(state.yardline_100 - yards_gained, 0)
    success_state = replace(state, down=1, ydstogo=10, yardline_100=new_yard)

    # If the offense reaches or crosses the goal line, award TD and flip possession
    if new_yard <= 0:
        success_state = replace(
            success_state,
            home_score=state.home_score + (7 if state.posteam == state.home_team else 0),
            away_score=state.away_score + (7 if state.posteam == state.away_team else 0),
        )
        success_state = flip_possession(success_state, new_yardline_100=75)  # touchback after score

    success_wp = predict_wp(model, success_state)

    # Failure state: turnover on downs at same spot, possession flips
    fail_yard = max(100 - state.yardline_100, 1)
    failure_state = flip_possession(state, new_yardline_100=fail_yard)
    failure_wp = predict_wp(model, failure_state)

    expected_wp = conv_prob * success_wp + (1 - conv_prob) * failure_wp
    return expected_wp, {
        "conv_prob": conv_prob,
        "success_wp": success_wp,
        "failure_wp": failure_wp,
    }


def simulate_field_goal(model, state: GameState, rate_models: RateModels | None = None) -> Tuple[float, Dict[str, float]]:
    fg_prob = estimate_fg_prob(state, rate_models)

    # Success: +3, possession flips on kickoff (assume touchback -> own 25)
    success_state = replace(
        state,
        home_score=state.home_score + (3 if state.posteam == state.home_team else 0),
        away_score=state.away_score + (3 if state.posteam == state.away_team else 0),
    )
    success_state = flip_possession(success_state, new_yardline_100=75)
    success_wp = predict_wp(model, success_state)

    # Miss: turnover at spot, possession flips
    miss_yard = max(100 - state.yardline_100, 1)
    miss_state = flip_possession(state, new_yardline_100=miss_yard)
    miss_wp = predict_wp(model, miss_state)

    expected_wp = fg_prob * success_wp + (1 - fg_prob) * miss_wp
    return expected_wp, {"fg_prob": fg_prob, "make_wp": success_wp, "miss_wp": miss_wp}


def simulate_punt(model, state: GameState, rate_models: RateModels | None = None) -> Tuple[float, Dict[str, float]]:
    net = estimate_punt_net_yards(state, rate_models)
    landing_from_kicking = max(state.yardline_100 - net, 0)
    new_yard_for_receiving = max(100 - landing_from_kicking, 1)
    punt_state = flip_possession(state, new_yardline_100=new_yard_for_receiving)
    punt_wp = predict_wp(model, punt_state)
    return punt_wp, {"net_yards": net}


def evaluate_fourth_down(model, state: GameState, rate_models: RateModels | None = None) -> List[Dict[str, float]]:
    """Evaluate go/punt/FG and return ranked decisions."""
    _validate_state(state)
    start = time.perf_counter()
    go_wp, go_meta = simulate_go_for_it(model, state, rate_models)
    fg_wp, fg_meta = simulate_field_goal(model, state, rate_models)
    punt_wp, punt_meta = simulate_punt(model, state, rate_models)

    decisions = [
        {"decision": "go for it", "wp": go_wp, **go_meta},
        {"decision": "field goal", "wp": fg_wp, **fg_meta},
        {"decision": "punt", "wp": punt_wp, **punt_meta},
    ]
    decisions.sort(key=lambda d: d["wp"], reverse=True)
    elapsed_ms = (time.perf_counter() - start) * 1000
    for d in decisions:
        d["sim_ms"] = elapsed_ms
    return decisions


def recommend_decision(model, state: GameState, rate_models: RateModels | None = None) -> Dict[str, float]:
    """Convenience wrapper to get the top recommendation."""
    return evaluate_fourth_down(model, state, rate_models)[0]


def batch_recommendations(model, states: Iterable[GameState]) -> List[Dict[str, float]]:
    """Evaluate a batch of states; returns list aligned with input order."""
    return [recommend_decision(model, s) for s in states]


def _validate_state(state: GameState) -> None:
    if not 1 <= state.down <= 4:
        raise ValueError(f"Down must be 1-4, got {state.down}")
    if state.ydstogo <= 0:
        raise ValueError(f"Yards to go must be > 0, got {state.ydstogo}")
    if not 1 <= state.yardline_100 <= 99:
        raise ValueError(f"yardline_100 must be 1-99, got {state.yardline_100}")
