import pandas as pd

from src.features.build_dataset import engineer_features


def test_engineer_features_basic():
    df = pd.DataFrame(
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
            }
        ]
    )

    X, y = engineer_features(df)
    assert list(X.columns) == [
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
        "is_home_offense",
        "is_home_defense",
        "goal_to_go",
        "two_min_drill",
        "timeouts_diff",
        "log_ydstogo",
    ]
    assert y.iloc[0] == 1
    assert X.iloc[0]["is_home_offense"] == 1
    assert X.iloc[0]["is_home_defense"] == 0
    assert X.iloc[0]["goal_to_go"] in (0, 1)
