"""Evaluate saved models on a few illustrative fourth-down scenarios."""

from __future__ import annotations

from pathlib import Path

from src.sim.fourth_down import (
    GameState,
    evaluate_fourth_down,
    load_model,
    load_rate_models,
)


def scenarios():
    """Sample scenarios. Adjust to include real-game moments for demos."""
    return [
        # Classic midfield 4th-and-1 early in game
        GameState(
            home_team="KC",
            away_team="BUF",
            posteam="KC",
            defteam="BUF",
            yardline_100=50,
            down=4,
            ydstogo=1,
            quarter_seconds_remaining=600,
            half_seconds_remaining=1800,
            game_seconds_remaining=1800,
            home_score=14,
            away_score=10,
            posteam_timeouts=3,
            defteam_timeouts=3,
        ),
        # High-leverage late-game 4th-and-2 in opponent territory
        GameState(
            home_team="PHI",
            away_team="DAL",
            posteam="PHI",
            defteam="DAL",
            yardline_100=35,
            down=4,
            ydstogo=2,
            quarter_seconds_remaining=90,
            half_seconds_remaining=90,
            game_seconds_remaining=90,
            home_score=24,
            away_score=27,
            posteam_timeouts=2,
            defteam_timeouts=1,
        ),
        # Backed-up 4th-and-5 with a lead in Q3
        GameState(
            home_team="SF",
            away_team="SEA",
            posteam="SF",
            defteam="SEA",
            yardline_100=20,
            down=4,
            ydstogo=5,
            quarter_seconds_remaining=300,
            half_seconds_remaining=900,
            game_seconds_remaining=1500,
            home_score=17,
            away_score=13,
            posteam_timeouts=3,
            defteam_timeouts=3,
        ),
    ]


def main(model_path: Path | str = "models/wp_model.joblib", rate_dir: Path | str = "models/rates") -> None:
    wp_model = load_model(model_path)
    rate_models = load_rate_models(rate_dir)

    for idx, state in enumerate(scenarios(), start=1):
        print(f"\nScenario {idx}: {state.posteam} vs {state.defteam} | 4th & {state.ydstogo} at {state.yardline_100}")
        decisions = evaluate_fourth_down(wp_model, state, rate_models=rate_models)
        for d in decisions:
            wp_pct = d['wp'] * 100
            extra = []
            if "conv_prob" in d and d["conv_prob"] is not None:
                extra.append(f"conv={d['conv_prob']*100:.0f}%")
            if "fg_prob" in d and d["fg_prob"] is not None:
                extra.append(f"fg%={d['fg_prob']*100:.0f}%")
            if "net_yards" in d and d["net_yards"] is not None:
                extra.append(f"net={d['net_yards']:.1f}")
            extra_str = " | " + ", ".join(extra) if extra else ""
            print(f"  - {d['decision']:<10} WP={wp_pct:5.1f}%{extra_str}")


if __name__ == "__main__":
    main()
