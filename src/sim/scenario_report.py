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
    """Real game situations for portfolio examples."""
    return [
        Scenario(
            name="Patriots vs Colts (Week 10, 2009) — 4th & 2 at own 28, 2:08 Q4, up 6",
            state=GameState(
                home_team="IND",
                away_team="NE",
                posteam="NE",
                defteam="IND",
                yardline_100=72,  # own 28 -> 72 yards to go
                down=4,
                ydstogo=2,
                quarter_seconds_remaining=128,
                half_seconds_remaining=128,
                game_seconds_remaining=128,
                home_score=34,
                away_score=27,
                posteam_timeouts=1,
                defteam_timeouts=3,
            ),
            actual_decision="go for it",
            note="Belichick 4th-and-2; failed; Colts won.",
        ),
        Scenario(
            name="Packers vs Buccaneers (NFC Champ 2020) — 4th & Goal at 8, 2:09 Q4, down 8",
            state=GameState(
                home_team="GB",
                away_team="TB",
                posteam="GB",
                defteam="TB",
                yardline_100=8,
                down=4,
                ydstogo=8,
                quarter_seconds_remaining=129,
                half_seconds_remaining=129,
                game_seconds_remaining=129,
                home_score=20,
                away_score=28,
                posteam_timeouts=3,
                defteam_timeouts=3,
            ),
            actual_decision="field goal",
            note="GB kicked FG down 8; criticized.",
        ),
        Scenario(
            name="Eagles vs Patriots (SB LII, 2018) — 4th & Goal at 1, 0:38 Q2",
            state=GameState(
                home_team="NE",
                away_team="PHI",
                posteam="PHI",
                defteam="NE",
                yardline_100=1,
                down=4,
                ydstogo=1,
                quarter_seconds_remaining=38,
                half_seconds_remaining=38,
                game_seconds_remaining=38 + 1800,  # end Q2
                home_score=12,
                away_score=15,
                posteam_timeouts=0,
                defteam_timeouts=0,
            ),
            actual_decision="go for it",
            note="Philly Special; TD.",
        ),
        Scenario(
            name="Lions vs Chargers (Week 10, 2023) — 4th & 2 at LAC 26, 1:47 Q4, tied",
            state=GameState(
                home_team="LAC",
                away_team="DET",
                posteam="DET",
                defteam="LAC",
                yardline_100=26,
                down=4,
                ydstogo=2,
                quarter_seconds_remaining=107,
                half_seconds_remaining=107,
                game_seconds_remaining=107,
                home_score=38,
                away_score=38,
                posteam_timeouts=3,
                defteam_timeouts=2,
            ),
            actual_decision="go for it",
            note="Converted; kicked GW FG.",
        ),
        Scenario(
            name="Chargers vs Browns (Week 5, 2022) — 4th & 2 at own 46, 1:14 Q4, up 2",
            state=GameState(
                home_team="CLE",
                away_team="LAC",
                posteam="LAC",
                defteam="CLE",
                yardline_100=54,  # own 46
                down=4,
                ydstogo=2,
                quarter_seconds_remaining=74,
                half_seconds_remaining=74,
                game_seconds_remaining=74,
                home_score=28,
                away_score=30,
                posteam_timeouts=3,
                defteam_timeouts=2,
            ),
            actual_decision="go for it",
            note="Failed; still won.",
        ),
        Scenario(
            name="Ravens vs Steelers (Week 13, 2021) — 4th & Goal at 2 (2-pt) 0:12 Q4, down 1",
            state=GameState(
                home_team="PIT",
                away_team="BAL",
                posteam="BAL",
                defteam="PIT",
                yardline_100=2,
                down=4,
                ydstogo=2,
                quarter_seconds_remaining=12,
                half_seconds_remaining=12,
                game_seconds_remaining=12,
                home_score=20,
                away_score=19,
                posteam_timeouts=0,
                defteam_timeouts=1,
            ),
            actual_decision="go for it",
            note="Harbaugh went for 2 to avoid OT; failed.",
        ),
        Scenario(
            name="Bills vs Chiefs (AFC Divisional 2021) — 4th & 4 at KC 44, 2:00 Q4, down 5",
            state=GameState(
                home_team="KC",
                away_team="BUF",
                posteam="BUF",
                defteam="KC",
                yardline_100=44,
                down=4,
                ydstogo=4,
                quarter_seconds_remaining=120,
                half_seconds_remaining=120,
                game_seconds_remaining=120,
                home_score=26,
                away_score=21,
                posteam_timeouts=3,
                defteam_timeouts=3,
            ),
            actual_decision="go for it",
            note="Converted; led to TD in :49 drive.",
        ),
        Scenario(
            name="Eagles vs 49ers (NFC Champ 2022) — 4th & 3 at SF 35, 10:21 Q1",
            state=GameState(
                home_team="PHI",
                away_team="SF",
                posteam="PHI",
                defteam="SF",
                yardline_100=35,
                down=4,
                ydstogo=3,
                quarter_seconds_remaining=621,
                half_seconds_remaining=1421,
                game_seconds_remaining=2221,
                home_score=0,
                away_score=0,
                posteam_timeouts=3,
                defteam_timeouts=3,
            ),
            actual_decision="go for it",
            note="Converted (controversial catch); led to TD.",
        ),
    ]


def run_report(model_path: Path | str = "models/wp_model.joblib", rate_dir: Path | str = "models/rates"):
    wp_model = load_model(model_path)
    rate_models = load_rate_models(rate_dir)

    rows = []
    for scenario in sample_scenarios():
        decisions = evaluate_fourth_down(wp_model, scenario.state, rate_models=rate_models)
        recommended = decisions[0]["decision"]
        rec_wp = decisions[0]["wp"]
        actual_wp = next((d["wp"] for d in decisions if d["decision"] == scenario.actual_decision), None)
        rows.append(
            {
                "scenario": scenario.name,
                "recommended": recommended,
                "recommended_wp": f"{rec_wp*100:.1f}%",
                "actual_wp": f"{actual_wp*100:.1f}%" if actual_wp is not None else "n/a",
                "delta_pp": f"{(rec_wp - actual_wp)*100:+.1f} pp" if actual_wp is not None else "n/a",
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
            f"actual={row['actual_decision']} ({row['actual_wp']}) [{status}] delta={row['delta_pp']} {row['note']}"
        )


def main():
    run_report()


if __name__ == "__main__":
    main()
