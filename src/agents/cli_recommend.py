"""CLI shortcut to return JSON recommendation and rationale."""

from __future__ import annotations

import argparse
import json

from src.llm.explain import explain_decision
from src.sim.fourth_down import (
    GameState,
    decision_intervals,
    evaluate_fourth_down,
    load_model,
    load_rate_models,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLI recommender (JSON).")
    parser.add_argument("--home", required=True)
    parser.add_argument("--away", required=True)
    parser.add_argument("--posteam", required=True)
    parser.add_argument("--yardline", type=float, required=True, help="1-99 yards to opponent end zone.")
    parser.add_argument("--down", type=int, default=4, choices=[1, 2, 3, 4])
    parser.add_argument("--ydstogo", type=float, required=True, help=">0 yards to first down.")
    parser.add_argument("--home-score", type=int, required=True, help=">=0")
    parser.add_argument("--away-score", type=int, required=True, help=">=0")
    parser.add_argument("--game-seconds", type=float, required=True, help="0-3600 seconds remaining.")
    parser.add_argument("--home-timeouts", type=int, default=3, choices=[0, 1, 2, 3])
    parser.add_argument("--away-timeouts", type=int, default=3, choices=[0, 1, 2, 3])
    parser.add_argument("--model-path", default="models/wp_model.joblib")
    parser.add_argument("--rate-dir", default="models/rates")
    parser.add_argument("--use-priors", action="store_true", help="Apply team priors if available.")
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
    model = load_model(args.model_path)
    rate_models = load_rate_models(args.rate_dir)
    if not args.use_priors and rate_models:
        rate_models.team_priors = None
    decisions = evaluate_fourth_down(model, gs, rate_models=rate_models)
    intervals = decision_intervals(model, gs, rate_models=rate_models, n_samples=200, alpha=0.1)
    best = decisions[0]
    rationale = explain_decision(best, gs)
    print(
        json.dumps(
            {
                "decision": best["decision"],
                "wp": best["wp"],
                "latency_ms": best.get("sim_ms"),
                "intervals": intervals.get(best["decision"]),
                "rationale": rationale,
                "decisions": decisions,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
