"""CLI entry to run the LangGraph decision workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint

from src.agents.langgraph_flow import as_dict, run_decision
from src.sim.fourth_down import GameState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LangGraph fourth-down decision agent.")
    parser.add_argument("--home", required=True, help="Home team code.")
    parser.add_argument("--away", required=True, help="Away team code.")
    parser.add_argument("--posteam", required=True, help="Possessing team code.")
    parser.add_argument("--yardline", type=float, required=True, help="Yards to opponent end zone (1-99).")
    parser.add_argument("--down", type=int, default=4, help="Down (1-4).")
    parser.add_argument("--ydstogo", type=float, required=True, help="Yards to go for first down.")
    parser.add_argument("--home-score", type=int, required=True)
    parser.add_argument("--away-score", type=int, required=True)
    parser.add_argument("--game-seconds", type=float, required=True, help="Seconds remaining in game.")
    parser.add_argument("--home-timeouts", type=int, default=3)
    parser.add_argument("--away-timeouts", type=int, default=3)
    parser.add_argument("--model-path", default="models/wp_model.joblib")
    parser.add_argument("--rate-dir", default="models/rates")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    defense = args.away if args.posteam == args.home else args.home
    gs = GameState(
        home_team=args.home,
        away_team=args.away,
        posteam=args.posteam,
        defteam=defense,
        yardline_100=args.yardline,
        down=args.down,
        ydstogo=args.ydstogo,
        quarter_seconds_remaining=min(args.game_seconds, 900),
        half_seconds_remaining=min(args.game_seconds, 1800),
        game_seconds_remaining=args.game_seconds,
        home_score=args.home_score,
        away_score=args.away_score,
        posteam_timeouts=args.home_timeouts if args.posteam == args.home else args.away_timeouts,
        defteam_timeouts=args.away_timeouts if args.posteam == args.home else args.home_timeouts,
    )
    result = run_decision(gs, model_path=Path(args.model_path), rate_dir=Path(args.rate_dir))
    pprint(as_dict(result))
    if result.get("explanation"):
        print("\nExplanation:", result["explanation"])


if __name__ == "__main__":
    main()
