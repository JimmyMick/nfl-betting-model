"""Interactive dashboard for the NFL win-probability model.

Two tabs over the validated pipeline:

* **Weekly preview** — a thin shell over ``predict.predict_week``: the
  model-vs-market table, the biggest disagreements, and an edge chart.
* **Season tracker** — a thin shell over ``grade.grade_season``: season-to-date
  record and calibration vs the market, plus a running-accuracy ticker.

A *probability/preview* tool, not a betting tip sheet (see predict.py).

Run with:
    uv run streamlit run dashboard.py
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from predict import predict_week, _prob_str
from grade import grade_season, weekly_summary, _calibration, _record

st.set_page_config(page_title="NFL model", page_icon="🏈", layout="wide")

CURRENT_SEASON = 2026


@st.cache_data(show_spinner=False)
def _predict(season: int, week: int, train_start: int, kind: str) -> pd.DataFrame:
    """Cached wrapper: training is expensive, so memoize per slate+model."""
    target = predict_week(season, week, train_start, kind)
    keep = [
        "home_team", "away_team", "model_home_prob", "market_home_prob",
        "edge", "driver", "home_win",
    ]
    return target[keep].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _grade(season: int, through_week: int, train_start: int, kind: str) -> pd.DataFrame:
    """Cached wrapper around the season grader."""
    s = grade_season(season, through_week, train_start, kind)
    keep = [
        "week", "home_team", "away_team", "model_home_prob", "market_home_prob",
        "home_win", "winner", "model_pick", "model_correct", "market_correct",
    ]
    return s[keep].reset_index(drop=True)


def _favoured(row: pd.Series) -> str:
    """Team the model's edge points to (the side it likes more than the market)."""
    return row["home_team"] if row["edge"] > 0 else row["away_team"]


def _display_table(df: pd.DataFrame, graded: bool) -> pd.DataFrame:
    """Format the raw frame into the human-readable preview table."""
    rows = []
    for _, r in df.iterrows():
        side = _favoured(r)
        row = {
            "Matchup": f"{r['away_team']} @ {r['home_team']}",
            "Model": _prob_str(r["home_team"], r["away_team"], r["model_home_prob"]),
            "Market": _prob_str(r["home_team"], r["away_team"], r["market_home_prob"]),
            "Edge": f"{side} +{abs(r['edge']):.0%}",
            "Key driver": r["driver"],
        }
        if graded:
            if pd.isna(r["home_win"]):
                row["Result"] = "—"
            else:
                winner = r["home_team"] if r["home_win"] == 1 else r["away_team"]
                pick = r["home_team"] if r["model_home_prob"] >= 0.5 else r["away_team"]
                row["Result"] = f"{winner} {'✓' if winner == pick else '✗'}"
        rows.append(row)
    return pd.DataFrame(rows)


def render_preview(season: int, week: int, train_start: int, kind: str) -> None:
    with st.spinner(f"Training on {train_start}–{season - 1} and predicting the slate…"):
        try:
            df = _predict(season, week, train_start, kind)
        except SystemExit as e:
            st.error(str(e) or f"No games found for {season} week {week}.")
            return

    graded = df["home_win"].notna().any()
    by_edge = df.reindex(df["edge"].abs().sort_values(ascending=False).index)

    st.subheader("Biggest model-vs-market disagreements")
    for col, (_, r) in zip(st.columns(3), by_edge.head(3).iterrows()):
        side = _favoured(r)
        col.metric(label=f"{r['away_team']} @ {r['home_team']}",
                   value=f"{side} +{abs(r['edge']):.0%}",
                   delta=r["driver"], delta_color="off")

    if graded:
        played = df[df["home_win"].notna()]
        picks = (played["model_home_prob"] >= 0.5).astype(int)
        mkt_picks = (played["market_home_prob"] >= 0.5).astype(int)
        acc = (picks == played["home_win"]).mean()
        mkt_acc = (mkt_picks == played["home_win"]).mean()
        c1, c2, c3 = st.columns(3)
        c1.metric("Model straight-up", f"{acc:.0%}",
                  f"{int((picks == played['home_win']).sum())}/{len(played)}")
        c2.metric("Market straight-up", f"{mkt_acc:.0%}")
        c3.metric("vs market", f"{acc - mkt_acc:+.0%}")

    st.subheader("Slate")
    st.dataframe(_display_table(by_edge, graded), width="stretch", hide_index=True)

    st.subheader("Edge by game (model minus market, toward favoured side)")
    chart_df = df.copy()
    chart_df["Matchup"] = chart_df["away_team"] + " @ " + chart_df["home_team"]
    chart_df["Favoured"] = chart_df.apply(_favoured, axis=1)
    chart_df["EdgePct"] = chart_df["edge"].abs() * 100
    chart = (
        alt.Chart(chart_df).mark_bar().encode(
            x=alt.X("EdgePct:Q", title="Edge vs market (%)"),
            y=alt.Y("Matchup:N", sort="-x", title=None),
            color=alt.Color("EdgePct:Q", scale=alt.Scale(scheme="blues"), legend=None),
            tooltip=["Matchup", "Favoured",
                     alt.Tooltip("EdgePct:Q", title="Edge %", format=".1f"),
                     alt.Tooltip("driver:N", title="Key driver")],
        ).properties(height=max(280, 26 * len(chart_df)))
    )
    st.altair_chart(chart, width="stretch")
    st.caption("Probability/preview tool. Model edges reflect disagreement with the "
               "closing line, not a profitable betting signal (moneyline is efficient).")


def render_tracker(season: int, through_week: int, train_start: int, kind: str) -> None:
    with st.spinner(f"Training on pre-{season} seasons and grading {season}…"):
        try:
            s = _grade(season, through_week, train_start, kind)
        except SystemExit as e:
            st.error(str(e) or f"Nothing to grade for {season}.")
            return

    m_acc = s["model_correct"].mean()
    k_acc = s["market_correct"].mean()
    m_ll, m_br, k_ll, k_br = _calibration(s)

    st.subheader(f"Season-to-date — through Week {int(s['week'].max())}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Model straight-up", _record(s["model_correct"]))
    c2.metric("Market straight-up", _record(s["market_correct"]))
    c3.metric("vs market", f"{m_acc - k_acc:+.0%}")
    c4, c5 = st.columns(2)
    c4.metric("Model calibration", f"logloss {m_ll:.3f}", f"Brier {m_br:.3f}",
              delta_color="off")
    c5.metric("Market calibration", f"logloss {k_ll:.3f}", f"Brier {k_br:.3f}",
              delta_color="off")

    # Running-accuracy ticker: cumulative model vs market accuracy by week.
    st.subheader("Accuracy ticker (cumulative)")
    wk = s.sort_values(["week"]).copy()
    wk["Model"] = wk["model_correct"].expanding().mean()
    wk["Market"] = wk["market_correct"].expanding().mean()
    cum = wk.groupby("week")[["Model", "Market"]].last().reset_index()
    long = cum.melt("week", var_name="Series", value_name="Accuracy")
    line = (
        alt.Chart(long).mark_line(point=True).encode(
            x=alt.X("week:O", title="Week"),
            y=alt.Y("Accuracy:Q", scale=alt.Scale(zero=False),
                    axis=alt.Axis(format="%")),
            color=alt.Color("Series:N", scale=alt.Scale(
                domain=["Model", "Market"], range=["#1f77b4", "#999999"])),
            tooltip=["week", "Series", alt.Tooltip("Accuracy:Q", format=".1%")],
        ).properties(height=320)
    )
    st.altair_chart(line, width="stretch")

    st.subheader("Week-by-week")
    st.dataframe(weekly_summary(s), width="stretch", hide_index=True)

    # This week's game-by-game grades.
    last = s[s["week"] == s["week"].max()]
    rows = [{
        "Matchup": f"{r['away_team']} @ {r['home_team']}",
        "Model pick": f"{r['model_pick']} {max(r['model_home_prob'], 1 - r['model_home_prob']):.0%}",
        "Result": f"{r['winner']} {'✓' if r['model_correct'] else '✗'}",
    } for _, r in last.iterrows()]
    st.subheader(f"Week {int(s['week'].max())} results")
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption("Scorekeeping companion to the weekly preview. Market-grade "
               "calibration is expected — the model is a forecaster, not a beater.")


# ── Sidebar (shared controls) ─────────────────────────────────────────────────
st.sidebar.title("🏈 NFL model")
season = st.sidebar.selectbox("Season", list(range(CURRENT_SEASON, 2009, -1)), index=0)
kind = st.sidebar.radio(
    "Model", ["logistic", "gbm"], index=0,
    help="logistic = saner tail probabilities (preview default); "
         "gbm = marginally better aggregate calibration, uglier tails",
)
train_start = st.sidebar.slider("Train from season", 2002, season - 1, 2010)
st.sidebar.caption(
    "Isotonic-calibrated full-feature model, trained on every season before the "
    "target and applied to strictly pre-game features. No picks, no EV claims."
)

preview_tab, tracker_tab = st.tabs(["Weekly preview", "Season tracker"])

with preview_tab:
    st.title(f"Preview — {season}")
    week = st.number_input("Week", min_value=1, max_value=22, value=1, step=1)
    if st.button("Run preview", type="primary", key="run_preview"):
        render_preview(int(season), int(week), int(train_start), kind)
    else:
        st.info("Pick a week and **Run preview**. First run for a slate trains "
                "the model (~30–60s); results are cached.")

with tracker_tab:
    st.title(f"Season tracker — {season}")
    through = st.number_input("Through week", min_value=1, max_value=22, value=1,
                              step=1, key="through_week")
    if st.button("Run tracker", type="primary", key="run_tracker"):
        render_tracker(int(season), int(through), int(train_start), kind)
    else:
        st.info("Pick a completed week and **Run tracker** for the season-to-date "
                "record, calibration, and accuracy ticker vs the market.")
