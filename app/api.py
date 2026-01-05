"""Minimal FastAPI app exposing a /recommend endpoint."""

from __future__ import annotations

from dataclasses import asdict
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.llm.explain import explain_decision
from src.sim.fourth_down import GameState, evaluate_fourth_down, load_model, load_rate_models

app = FastAPI(title="NFL Game Management API", version="0.1.0")

WP_MODEL = load_model()
RATE_MODELS = load_rate_models()


class RecommendRequest(BaseModel):
    home_team: str
    away_team: str
    posteam: str
    yardline_100: float = Field(..., gt=0, lt=100)
    down: int = Field(..., ge=1, le=4)
    ydstogo: float = Field(..., gt=0)
    game_seconds_remaining: float = Field(..., ge=0, le=3600)
    home_score: int = Field(..., ge=0)
    away_score: int = Field(..., ge=0)
    home_timeouts: int = Field(3, ge=0, le=3)
    away_timeouts: int = Field(3, ge=0, le=3)


class RecommendResponse(BaseModel):
    decision: str
    wp: float
    rationale: str
    decisions: Optional[list]
    latency_ms: Optional[float]


@app.post("/recommend", response_model=RecommendResponse)
def recommend(body: RecommendRequest):
    defense = body.away_team if body.posteam == body.home_team else body.home_team
    gs = GameState(
        home_team=body.home_team,
        away_team=body.away_team,
        posteam=body.posteam,
        defteam=defense,
        yardline_100=body.yardline_100,
        down=body.down,
        ydstogo=body.ydstogo,
        quarter_seconds_remaining=min(body.game_seconds_remaining, 900),
        half_seconds_remaining=min(body.game_seconds_remaining, 1800),
        game_seconds_remaining=body.game_seconds_remaining,
        home_score=body.home_score,
        away_score=body.away_score,
        posteam_timeouts=body.home_timeouts if body.posteam == body.home_team else body.away_timeouts,
        defteam_timeouts=body.away_timeouts if body.posteam == body.home_team else body.home_timeouts,
    )

    decisions = evaluate_fourth_down(WP_MODEL, gs, rate_models=RATE_MODELS)
    best = decisions[0]
    rationale = explain_decision(best, gs)
    return RecommendResponse(
        decision=best["decision"],
        wp=best["wp"],
        rationale=rationale,
        decisions=[{k: v for k, v in d.items()} for d in decisions],
        latency_ms=best.get("sim_ms"),
    )
