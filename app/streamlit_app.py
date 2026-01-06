"""Streamlit UI for fourth-down decision support."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

VERSION_FILE = PROJECT_ROOT / "VERSION"
MODEL_VERSION = VERSION_FILE.read_text().strip() if VERSION_FILE.exists() else "0.0.0"

from src.llm.explain import explain_decision  # noqa: E402
from src.sim.fourth_down import GameState, evaluate_fourth_down, load_model, load_rate_models  # noqa: E402
from src.sim.fourth_down import decision_intervals  # noqa: E402


class Preset:
    def __init__(self, name: str, state: GameState, actual_decision: str, note: str = ""):
        self.name = name
        self.state = state
        self.actual_decision = actual_decision
        self.note = note


PRESETS = {
    "KC vs BUF — 4th & 1 @ 50 (Q2)": Preset(
        name="KC vs BUF — 4th & 1 @ 50 (Q2)",
        actual_decision="go for it",
        note="Illustrative; replace with actual WP if known.",
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
    ),
    "GB vs TB — NFC Champ 2020 — 4th & Goal @ 8 (Q4)": Preset(
        name="GB vs TB — NFC Champ 2020 — 4th & Goal @ 8 (Q4)",
        actual_decision="field goal",
        note="Packers kicked FG down 8; criticized call.",
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
    ),
    "Lions vs Chargers 2023 — 4th & 2 @ 26 (Q4)": Preset(
        name="Lions vs Chargers 2023 — 4th & 2 @ 26 (Q4)",
        actual_decision="go for it",
        note="Campbell went; converted, then kicked GW FG.",
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
    ),
    "Chargers vs Browns 2022 — 4th & 2 @ own 46 (Q4)": Preset(
        name="Chargers vs Browns 2022 — 4th & 2 @ own 46 (Q4)",
        actual_decision="go for it",
        note="Staley went; failed; still won.",
        state=GameState(
            home_team="CLE",
            away_team="LAC",
            posteam="LAC",
            defteam="CLE",
            yardline_100=54,  # own 46 -> 54 yards to opponent end zone
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
    ),
}


@st.cache_resource
def get_model(model_path: str = "models/wp_model.joblib"):
    return load_model(model_path)


def derive_clocks(game_seconds_remaining: float) -> tuple[float, float, float]:
    g = max(game_seconds_remaining, 0)
    half_seconds = min(g, 1800)
    quarter_seconds = g % 900 or min(g, 900)
    return quarter_seconds, half_seconds, g


@st.cache_resource
def get_rate_models():
    return load_rate_models()


def validate_inputs(yardline: int, ydstogo: float, down: int) -> tuple[bool, str]:
    if not 1 <= yardline <= 99:
        return False, "Yardline must be between 1 and 99 (distance to end zone)."
    if ydstogo <= 0:
        return False, "Yards to go must be greater than 0."
    if not 1 <= down <= 4:
        return False, "Down must be between 1 and 4."
    return True, ""


def main() -> None:
    st.title("NFL Game Management AI — Fourth Down")
    st.caption(f"Model version: {MODEL_VERSION}")
    st.caption("Heuristic simulator powered by a trained win-probability model.")

    preset = st.selectbox("Load a preset scenario", options=["(none)"] + list(PRESETS.keys()))
    if preset != "(none)":
        selected = PRESETS[preset]
        gs = selected.state
        home_default, away_default = gs.home_team, gs.away_team
        home_score_default, away_score_default = gs.home_score, gs.away_score
        home_to_default = gs.posteam_timeouts if gs.posteam == gs.home_team else gs.defteam_timeouts
        away_to_default = gs.posteam_timeouts if gs.posteam == gs.away_team else gs.defteam_timeouts
        pos_default = gs.posteam
        yard_default = int(gs.yardline_100)
        down_default = int(gs.down)
        ytg_default = float(gs.ydstogo)
        game_sec_default = int(gs.game_seconds_remaining)
    else:
        home_default, away_default = "HOME", "AWAY"
        home_score_default, away_score_default = 17, 14
        home_to_default, away_to_default = 3, 3
        pos_default = home_default
        yard_default = 45
        down_default = 4
        ytg_default = 4.0
        game_sec_default = 900

    col_home, col_away = st.columns(2)
    with col_home:
        home_team = st.text_input("Home team", value=home_default)
        home_score = st.number_input("Home score", min_value=0, value=home_score_default)
        home_timeouts = st.selectbox(
            "Home timeouts remaining", [0, 1, 2, 3], index=[0, 1, 2, 3].index(home_to_default) if preset != "(none)" else 3
        )
    with col_away:
        away_team = st.text_input("Away team", value=away_default)
        away_score = st.number_input("Away score", min_value=0, value=away_score_default)
        away_timeouts = st.selectbox(
            "Away timeouts remaining", [0, 1, 2, 3], index=[0, 1, 2, 3].index(away_to_default) if preset != "(none)" else 3
        )

    pos = st.selectbox("Possession", options=[home_team, away_team], index=[home_team, away_team].index(pos_default))
    defense = away_team if pos == home_team else home_team

    yardline = st.slider("Yards to opponent end zone (yardline_100)", min_value=1, max_value=99, value=yard_default)
    down = st.selectbox("Down", options=[1, 2, 3, 4], index=[1, 2, 3, 4].index(down_default))
    ydstogo = st.number_input("Yards to go", min_value=1.0, value=ytg_default, step=0.5)

    game_seconds_remaining = st.slider(
        "Game seconds remaining", min_value=0, max_value=3600, value=game_sec_default, step=30
    )
    qtr_sec, half_sec, game_sec = derive_clocks(game_seconds_remaining)

    st.write(
        f"Derived clocks — Quarter: {qtr_sec:.0f}s, Half: {half_sec:.0f}s, Game: {game_sec:.0f}s"
    )

    using_learned = False
    if st.button("Evaluate decisions", type="primary"):
        ok, msg = validate_inputs(yardline, ydstogo, down)
        if not ok:
            st.error(msg)
            return
        try:
            model = get_model()
        except FileNotFoundError as exc:
            st.error(f"{exc} — run training first.")
            return
        rate_models = get_rate_models()
        if not use_priors and rate_models:
            rate_models.team_priors = None
        using_learned = any(
            [
                rate_models.conversion_model is not None,
                rate_models.fg_model is not None,
                rate_models.punt_model is not None,
            ]
        )

        state = GameState(
            home_team=home_team,
            away_team=away_team,
            posteam=pos,
            defteam=defense,
            yardline_100=yardline,
            down=down,
            ydstogo=ydstogo,
            quarter_seconds_remaining=qtr_sec,
            half_seconds_remaining=half_sec,
            game_seconds_remaining=game_sec,
            home_score=home_score,
            away_score=away_score,
            posteam_timeouts=home_timeouts if pos == home_team else away_timeouts,
            defteam_timeouts=away_timeouts if pos == home_team else home_timeouts,
        )

        decisions = evaluate_fourth_down(model, state, rate_models=rate_models)
        intervals = decision_intervals(model, state, rate_models=rate_models, n_samples=200, alpha=0.1)
        table = pd.DataFrame(decisions)
        table["wp_pct"] = table["wp"] * 100
        st.subheader("Decisions (higher WP is better)")
        st.dataframe(table[["decision", "wp_pct", "conv_prob", "fg_prob", "net_yards"]], use_container_width=True)

        best = decisions[0]
        st.success(f"Recommended: **{best['decision']}** — projected home WP {best['wp']*100:.1f}%")
        st.write(explain_decision(best, state))
        if preset != "(none)":
            actual = selected.actual_decision
            actual_wp = next((d["wp"] for d in decisions if d["decision"] == actual), None)
            if actual_wp is not None:
                delta = (best["wp"] - actual_wp) * 100
                st.info(
                    f"Delta vs actual decision ({actual}): {delta:+.1f} pp WP"
                    + (f" — {selected.note}" if selected.note else "")
                )
        # Uncertainty range from Monte Carlo on decision outcomes
        interval = intervals.get(best["decision"])
        if interval:
            st.caption(
                f"Approximate WP range (10-90%): [{interval['low']*100:.1f}%, {interval['high']*100:.1f}%] "
                f"| mean {interval['mean']*100:.1f}%"
            )

    st.markdown("---")
    if using_learned:
        st.caption("Using learned conversion/FG/punt models when available; falling back to heuristics otherwise.")
    else:
        st.caption("Note: Probabilities use simple heuristics; train rate models for data-driven estimates.")

    st.markdown("### Sensitivity")
    with st.expander("WP sensitivity to ydstogo and timeouts"):
        deltas = [-1, 0, 1]
        rows = []
        for dy in deltas:
            adj_ydtogo = max(1.0, ydstogo + dy)
            adj_state = GameState(
                home_team=home_team,
                away_team=away_team,
                posteam=pos,
                defteam=defense,
                yardline_100=yardline,
                down=down,
                ydstogo=adj_ydtogo,
                quarter_seconds_remaining=qtr_sec,
                half_seconds_remaining=half_sec,
                game_seconds_remaining=game_sec,
                home_score=home_score,
                away_score=away_score,
                posteam_timeouts=home_timeouts if pos == home_team else away_timeouts,
                defteam_timeouts=away_timeouts if pos == home_team else home_timeouts,
            )
            dec = evaluate_fourth_down(model, adj_state, rate_models=rate_models)[0]
            rows.append({"ydstogo": adj_ydtogo, "decision": dec["decision"], "wp_pct": dec["wp"] * 100})

        # Timeout sensitivity: flip timeouts +/-1 within bounds
        to_rows = []
        for delta_to in deltas:
            adj_off_to = min(3, max(0, (home_timeouts if pos == home_team else away_timeouts) + delta_to))
            adj_def_to = min(3, max(0, (away_timeouts if pos == home_team else home_timeouts) + delta_to))
            adj_state = GameState(
                home_team=home_team,
                away_team=away_team,
                posteam=pos,
                defteam=defense,
                yardline_100=yardline,
                down=down,
                ydstogo=ydstogo,
                quarter_seconds_remaining=qtr_sec,
                half_seconds_remaining=half_sec,
                game_seconds_remaining=game_sec,
                home_score=home_score,
                away_score=away_score,
                posteam_timeouts=adj_off_to,
                defteam_timeouts=adj_def_to,
            )
            dec = evaluate_fourth_down(model, adj_state, rate_models=rate_models)[0]
            to_rows.append({"timeouts_delta": delta_to, "decision": dec["decision"], "wp_pct": dec["wp"] * 100})

        st.write("Yards to go sensitivity")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.write("Timeouts sensitivity (offense/defense +/-1)")
        st.dataframe(pd.DataFrame(to_rows), use_container_width=True)
        # Simple WP vs ydstogo curve around current state
        ytg_range = [max(1, ydstogo + delta) for delta in range(-3, 4)]
        chart_rows = []
        for ytg_val in sorted(set(ytg_range)):
            adj_state = GameState(
                home_team=home_team,
                away_team=away_team,
                posteam=pos,
                defteam=defense,
                yardline_100=yardline,
                down=down,
                ydstogo=ytg_val,
                quarter_seconds_remaining=qtr_sec,
                half_seconds_remaining=half_sec,
                game_seconds_remaining=game_sec,
                home_score=home_score,
                away_score=away_score,
                posteam_timeouts=home_timeouts if pos == home_team else away_timeouts,
                defteam_timeouts=away_timeouts if pos == home_team else home_timeouts,
            )
            dec = evaluate_fourth_down(model, adj_state, rate_models=rate_models)[0]
            chart_rows.append({"ydstogo": ytg_val, "wp_pct": dec["wp"] * 100})
        chart_df = pd.DataFrame(chart_rows).sort_values("ydstogo")
        st.line_chart(chart_df.set_index("ydstogo"), height=220)

    if rate_models and rate_models.team_priors:
        st.markdown("### Team priors")
        priors_df = pd.DataFrame(
            [
                {"team": team, "prior": val, "offense": val, "defense": -val}
                for team, val in rate_models.team_priors.items()
            ]
        ).sort_values("prior", ascending=False)
        st.dataframe(priors_df, use_container_width=True)

        st.markdown("#### Compare with vs without priors")
        state_for_compare = state
        with_priors = evaluate_fourth_down(model, state_for_compare, rate_models=rate_models)[0]
        without_priors_models = load_rate_models()
        if without_priors_models:
            without_priors_models.team_priors = None
        without_priors = evaluate_fourth_down(model, state_for_compare, rate_models=without_priors_models)[0]
        compare_df = pd.DataFrame(
            [
                {"mode": "with priors", "decision": with_priors["decision"], "wp_pct": with_priors["wp"] * 100},
                {"mode": "without priors", "decision": without_priors["decision"], "wp_pct": without_priors["wp"] * 100},
            ]
        )
        st.dataframe(compare_df, use_container_width=True)


if __name__ == "__main__":
    main()
    use_priors = st.checkbox("Apply team priors (if available)", value=True)
