from src.sim.fourth_down import (
    GameState,
    evaluate_fourth_down,
    estimate_fg_prob,
    estimate_go_conversion_prob,
    estimate_punt_net_yards,
)


class DummyModel:
    def __init__(self, wp: float = 0.5):
        self.wp = wp

    def predict_proba(self, X):
        # Return fixed probability
        import numpy as np

        return np.array([[1 - self.wp, self.wp] for _ in range(len(X))])


def sample_state() -> GameState:
    return GameState(
        home_team="HOME",
        away_team="AWAY",
        posteam="HOME",
        defteam="AWAY",
        yardline_100=50,
        down=4,
        ydstogo=2,
        quarter_seconds_remaining=900,
        half_seconds_remaining=1800,
        game_seconds_remaining=1800,
        home_score=10,
        away_score=7,
        posteam_timeouts=3,
        defteam_timeouts=3,
    )


def test_estimates_in_bounds():
    state = sample_state()
    assert 0 < estimate_go_conversion_prob(state) <= 1
    assert 0 < estimate_fg_prob(state) <= 1
    assert 10 < estimate_punt_net_yards(state) <= 50


def test_evaluate_fourth_down_orders_by_wp():
    model = DummyModel(wp=0.6)
    decisions = evaluate_fourth_down(model, sample_state())
    assert decisions[0]["decision"] in {"go for it", "field goal", "punt"}
    # Ensure sorted descending
    wps = [d["wp"] for d in decisions]
    assert wps == sorted(wps, reverse=True)
