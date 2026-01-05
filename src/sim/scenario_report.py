"""Compare model recommendations to tagged real/illustrative decisions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from src.sim.fourth_down import (
    GameState,
    evaluate_fourth_down,
    load_model,
    load_rate_models,
)


@dataclass
class Scenario:
    name: str
    state: GameState
    actual_decision: str
    note: str = ""


def sample_scenarios() -> List[Scenario]:
    """Swap in real historical situations as you research them."""
    return [
        Scenario(
            name="KC vs BUF, 4th & 1 at 50 (Q2)",
            state=GameState(
                home_team="KC",
                away_team="BUF",
                posteam="KC",
                defteam="BUF",
                yardline_100=50,
                down=4,
                ydstogo=1,
                quarter_seconds_remaining=420,
                half_seconds_remaining=1200,
                game_seconds_remaining=1800,
                home_score=14,
                away_score=10,
                posteam_timeouts=3,
                defteam_timeouts=3,
            ),
            actual_decision="go for it",
            note="Illustrative; swap with real play.",
        ),
        Scenario(
            name="PHI vs DAL, 4th & 2 at 35 (late Q4)",
            state=GameState(
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
            actual_decision="field goal",
            note="Illustrative; replace with actual game context.",
        ),
        Scenario(
            name="SF vs SEA, 4th & 5 at own 20 (Q3)",
            state=GameState(
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
            actual_decision="punt",
            note="Illustrative; replace with actual play.",
        ),
    ]


def run_report(model_path: Path | str = "models/wp_model.joblib", rate_dir: Path | str = "models/rates"):
    wp_model = load_model(model_path)
    rate_models = load_rate_models(rate_dir)

    rows = []
    for scenario in sample_scenarios():
        decisions = evaluate_fourth_down(wp_model, scenario.state, rate_models=rate_models)
        recommended = decisions[0]["decision"]
        rows.append(
            {
                "scenario": scenario.name,
                "recommended": recommended,
                "recommended_wp": f"{decisions[0]['wp']*100:.1f}%",
                "actual_decision": scenario.actual_decision,
                "aligned": recommended == scenario.actual_decision,
                "note": scenario.note,
            }
        )

    print("\nScenario Report:")
    for row in rows:
        status = "MATCH" if row["aligned"] else "DIFF"
        print(
            f"- {row['scenario']}: rec={row['recommended']} ({row['recommended_wp']}), "
            f"actual={row['actual_decision']} [{status}] {row['note']}"
        )


def main():
    run_report()


if __name__ == "__main__":
    main()
