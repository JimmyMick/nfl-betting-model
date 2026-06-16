"""Interactive weekly preview dashboard for the NFL win-probability model.

A thin Streamlit shell over ``predict.predict_week`` — pick a season/week/model
and browse the model-vs-market table, the biggest disagreements, and (for graded
weeks) how the model's straight-up picks fared against Vegas.

This is a *probability/preview* tool, not a betting tip sheet (see predict.py).

Run with:
    uv run streamlit run dashboard.py
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from predict import predict_week, _prob_str

st.set_page_config(page_title="NFL model — weekly preview", page_icon="🏈",
                   layout="wide")

CURRENT_SEASON = 2026


@st.cache_data(show_spinner=False)
def _predict(season: int, week: int, train_start: int, kind: str) -> pd.DataFrame:
    """Cached wrapper: training is expensive, so memoize per slate+model.

    Returns a plain DataFrame (cache-friendly) with the columns the UI needs.
    """
    target = predict_week(season, week, train_start, kind)
    keep = [
        "home_team", "away_team", "model_home_prob", "market_home_prob",
        "edge", "driver", "home_win",
    ]
    return target[keep].reset_index(drop=True)


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


# ── Sidebar controls ─────────────────────────────────────────────────────────
st.sidebar.title("🏈 Weekly preview")
season = st.sidebar.selectbox("Season", list(range(CURRENT_SEASON, 2009, -1)), index=0)
week = st.sidebar.number_input("Week", min_value=1, max_value=22, value=1, step=1)
kind = st.sidebar.radio(
    "Model", ["logistic", "gbm"], index=0,
    help="logistic = saner tail probabilities (preview default); "
         "gbm = marginally better aggregate calibration, uglier tails",
)
train_start = st.sidebar.slider("Train from season", 2002, season - 1, 2010)
go = st.sidebar.button("Run preview", type="primary", width="stretch")

st.sidebar.caption(
    "Trains the isotonic-calibrated full-feature model on every season before "
    "the target, then predicts the slate from strictly pre-game features. "
    "A preview/probability tool — no picks, no EV claims."
)

st.title(f"NFL model — {season} Week {int(week)}")

if not go:
    st.info("Pick a season and week in the sidebar, then **Run preview**. "
            "First run for a slate trains the model (~30–60s); results are cached.")
    st.stop()

with st.spinner(f"Training on {train_start}–{season - 1} and predicting the slate…"):
    try:
        df = _predict(int(season), int(week), int(train_start), kind)
    except SystemExit as e:
        st.error(str(e) or f"No games found for {season} week {int(week)}.")
        st.stop()

graded = df["home_win"].notna().any()

# ── Headline: biggest disagreements ──────────────────────────────────────────
st.subheader("Biggest model-vs-market disagreements")
top = df.reindex(df["edge"].abs().sort_values(ascending=False).index).head(3)
cols = st.columns(3)
for col, (_, r) in zip(cols, top.iterrows()):
    side = _favoured(r)
    col.metric(
        label=f"{r['away_team']} @ {r['home_team']}",
        value=f"{side} +{abs(r['edge']):.0%}",
        delta=r["driver"], delta_color="off",
    )

# ── Graded summary ───────────────────────────────────────────────────────────
if graded:
    played = df[df["home_win"].notna()]
    picks = (played["model_home_prob"] >= 0.5).astype(int)
    mkt_picks = (played["market_home_prob"] >= 0.5).astype(int)
    acc = (picks == played["home_win"]).mean()
    mkt_acc = (mkt_picks == played["home_win"]).mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("Model straight-up", f"{acc:.0%}", f"{int((picks == played['home_win']).sum())}/{len(played)}")
    c2.metric("Market straight-up", f"{mkt_acc:.0%}")
    c3.metric("vs market", f"{acc - mkt_acc:+.0%}")

# ── Main table ───────────────────────────────────────────────────────────────
st.subheader("Slate")
table = _display_table(
    df.reindex(df["edge"].abs().sort_values(ascending=False).index), graded,
)
st.dataframe(table, width="stretch", hide_index=True)

# ── Edge chart ───────────────────────────────────────────────────────────────
st.subheader("Edge by game (model minus market, toward favoured side)")
chart_df = df.copy()
chart_df["Matchup"] = chart_df["away_team"] + " @ " + chart_df["home_team"]
chart_df["Favoured"] = chart_df.apply(_favoured, axis=1)
chart_df["EdgePct"] = chart_df["edge"].abs() * 100
chart = (
    alt.Chart(chart_df)
    .mark_bar()
    .encode(
        x=alt.X("EdgePct:Q", title="Edge vs market (%)"),
        y=alt.Y("Matchup:N", sort="-x", title=None),
        color=alt.Color("EdgePct:Q", scale=alt.Scale(scheme="blues"),
                        legend=None),
        tooltip=["Matchup", "Favoured",
                 alt.Tooltip("EdgePct:Q", title="Edge %", format=".1f"),
                 alt.Tooltip("driver:N", title="Key driver")],
    )
    .properties(height=max(280, 26 * len(chart_df)))
)
st.altair_chart(chart, width="stretch")

st.caption("Probability/preview tool. Model edges reflect disagreement with the "
           "closing line, not a profitable betting signal (moneyline is efficient).")
