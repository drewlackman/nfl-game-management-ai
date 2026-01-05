"""Lightweight explainer for model-driven decisions.

This is a stub; swap with a real LLM call (OpenAI, Anthropic, etc.) if desired.
"""

from __future__ import annotations

from typing import Dict

from src.sim.fourth_down import GameState


def explain_decision(decision: Dict[str, float], state: GameState) -> str:
    """Generate a brief rationale string."""
    name = decision["decision"]
    wp_pct = decision["wp"] * 100
    if name == "go for it":
        conv = decision.get("conv_prob", 0) * 100
        return (
            f"Go for it yields {wp_pct:.1f}% home win chance with "
            f"{conv:.0f}% conversion odds; success keeps the drive alive near the "
            f"{state.yardline_100 - max(state.ydstogo, 1):.0f} yard line."
        )
    if name == "field goal":
        fg = decision.get("fg_prob", 0) * 100
        return (
            f"Field goal projects {wp_pct:.1f}% WP with a {fg:.0f}% make rate; "
            "on a make you add three and kick off, on a miss opponent takes over at the spot."
        )
    if name == "punt":
        net = decision.get("net_yards", 0)
        return (
            f"Punting nets ~{net:.0f} yards and sets WP at {wp_pct:.1f}%. "
            "Leverages field position to lower opponent scoring odds."
        )
    return f"{name} yields {wp_pct:.1f}% WP based on current heuristics."
