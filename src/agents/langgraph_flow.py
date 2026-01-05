"""LangGraph workflow for fourth-down decision-making."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, TypedDict

from langgraph.graph import END, StateGraph

from src.sim.fourth_down import (
    GameState,
    evaluate_fourth_down,
    load_model,
    load_rate_models,
)
from src.llm.explain import explain_decision


class DecisionState(TypedDict, total=False):
    game_state: GameState
    decisions: Any
    recommended: Dict[str, Any]
    explanation: str
    status: str
    error: str


def build_graph(model_path: Path | str = "models/wp_model.joblib", rate_dir: Path | str = "models/rates"):
    wp_model = load_model(model_path)
    rate_models = load_rate_models(rate_dir)

    def compute_decisions(state: DecisionState) -> DecisionState:
        try:
            gs = state["game_state"]
            decisions = evaluate_fourth_down(wp_model, gs, rate_models=rate_models)
            state["decisions"] = decisions
            state["recommended"] = decisions[0]
            state["status"] = "ok"
        except Exception as exc:  # pragma: no cover - defensive
            state["error"] = str(exc)
            state["status"] = "error"
        return state

    def add_explanation(state: DecisionState) -> DecisionState:
        if state.get("status") != "ok":
            return state
        try:
            state["explanation"] = explain_decision(state["recommended"], state["game_state"])
        except Exception:
            state["explanation"] = ""
        return state

    workflow = StateGraph(DecisionState)
    workflow.add_node("compute", compute_decisions)
    workflow.add_node("explain", add_explanation)

    workflow.set_entry_point("compute")
    workflow.add_edge("compute", "explain")
    workflow.add_edge("explain", END)
    return workflow.compile()


def run_decision(game_state: GameState, model_path: Path | str = "models/wp_model.joblib", rate_dir: Path | str = "models/rates"):
    graph = build_graph(model_path=model_path, rate_dir=rate_dir)
    result = graph.invoke({"game_state": game_state})
    return result


def as_dict(decision_state: DecisionState) -> Dict[str, Any]:
    """Serialize DecisionState to plain dict for printing/logging."""
    out = dict(decision_state)
    if "game_state" in out and isinstance(out["game_state"], GameState):
        out["game_state"] = asdict(out["game_state"])
    return out
