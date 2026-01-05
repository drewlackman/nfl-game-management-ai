import json
import os
from pathlib import Path

import pandas as pd

from src.features.build_dataset import engineer_features
from src.models.train_wp import build_pipeline
from src.sim.fourth_down import GameState, evaluate_fourth_down


def test_end_to_end_small(tmp_path, monkeypatch):
    # Create tiny synthetic dataset
    data = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "posteam": "HOME",
                "defteam": "AWAY",
                "home_team": "HOME",
                "away_team": "AWAY",
                "yardline_100": 50,
                "down": 4,
                "ydstogo": 2,
                "quarter_seconds_remaining": 120,
                "half_seconds_remaining": 600,
                "game_seconds_remaining": 1200,
                "score_differential": 3,
                "posteam_timeouts_remaining": 2,
                "defteam_timeouts_remaining": 3,
                "result": 7,
            },
            {
                "game_id": "g2",
                "posteam": "AWAY",
                "defteam": "HOME",
                "home_team": "HOME",
                "away_team": "AWAY",
                "yardline_100": 30,
                "down": 3,
                "ydstogo": 5,
                "quarter_seconds_remaining": 300,
                "half_seconds_remaining": 900,
                "game_seconds_remaining": 1800,
                "score_differential": -4,
                "posteam_timeouts_remaining": 3,
                "defteam_timeouts_remaining": 2,
                "result": -3,
            },
        ]
    )

    features, target = engineer_features(data)
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
    model = build_pipeline(categorical, numeric, calibrate=False, model_type="gboost")
    model.fit(features, target)

    # Simulate a state with the trained model
    state = GameState(
        home_team="HOME",
        away_team="AWAY",
        posteam="HOME",
        defteam="AWAY",
        yardline_100=45,
        down=4,
        ydstogo=2,
        quarter_seconds_remaining=90,
        half_seconds_remaining=300,
        game_seconds_remaining=900,
        home_score=10,
        away_score=7,
        posteam_timeouts=3,
        defteam_timeouts=2,
    )
    decisions = evaluate_fourth_down(model, state)
    assert decisions and decisions[0]["decision"] in {"go for it", "field goal", "punt"}
