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

import datetime as dt

import altair as alt
import pandas as pd
import streamlit as st
from fpdf import FPDF

from predict import predict_week, _prob_str
from grade import grade_season, weekly_summary, _calibration, _record
from nfl_betting_model import data, picks as picks_mod
from nfl_betting_model.roster import team_roster

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
        "game_id", "week", "home_team", "away_team", "model_home_prob",
        "market_home_prob", "home_win", "winner", "model_pick", "model_correct",
        "market_correct",
    ]
    return s[keep].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _teams(season: int) -> list[str]:
    """Teams on the season's schedule (cheap — schedule only), sorted."""
    games = data.load_games([season], include_unplayed=True)
    return sorted(pd.unique(games[["home_team", "away_team"]].values.ravel()).tolist())


@st.cache_data(show_spinner=False)
def _roster(team: str, season: int) -> pd.DataFrame:
    """Cached team roster (snap counts + Madden ratings)."""
    return team_roster(team, season)


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


def _ascii(s: object) -> str:
    """Latin-1-safe text for fpdf core fonts (drivers use a Unicode arrow)."""
    s = str(s).replace("→", "->").replace("–", "-").replace("—", "-")
    return s.encode("latin-1", "replace").decode("latin-1")


def _preview_pdf(season: int, week: int, kind: str, table: pd.DataFrame,
                 disagreements: list[str], acc_line: str | None) -> bytes:
    """Render the preview report (summary + sorted slate) to PDF bytes."""
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, _ascii(f"NFL Model - Preview: {season} Week {week}"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(110)
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    pdf.cell(0, 6, _ascii(f"{kind} model - generated {stamp} - "
                          "sorted by model win probability"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0)
    pdf.ln(2)

    if disagreements:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Biggest model-vs-market disagreements",
                 new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        for line in disagreements:
            pdf.cell(0, 5.5, _ascii(f"  - {line}"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
    if acc_line:
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, _ascii(acc_line), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Slate", new_x="LMARGIN", new_y="NEXT")

    cols = list(table.columns)
    weights = {"Matchup": 24, "Model": 16, "Market": 16, "Edge": 18,
               "Key driver": 42, "Result": 18}
    col_widths = tuple(weights.get(c, 20) for c in cols)
    pdf.set_font("Helvetica", "", 9)
    with pdf.table(col_widths=col_widths, text_align="LEFT",
                   first_row_as_headings=True) as t:
        t.row([_ascii(c) for c in cols])
        for _, r in table.iterrows():
            t.row([_ascii(r[c]) for c in cols])

    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(110)
    pdf.ln(2)
    pdf.multi_cell(0, 4, _ascii(
        "Probability/preview tool. Model edges reflect disagreement with the "
        "closing line, not a profitable betting signal (moneyline is efficient)."))
    return bytes(pdf.output())


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
    disagreements = []
    for col, (_, r) in zip(st.columns(3), by_edge.head(3).iterrows()):
        side = _favoured(r)
        col.metric(label=f"{r['away_team']} @ {r['home_team']}",
                   value=f"{side} +{abs(r['edge']):.0%}",
                   delta=r["driver"], delta_color="off")
        disagreements.append(
            f"{r['away_team']} @ {r['home_team']}: {side} +{abs(r['edge']):.0%} "
            f"({r['driver']})")

    acc_line = None
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
        acc_line = (f"Straight-up: model {acc:.0%} vs market {mkt_acc:.0%} "
                    f"({acc - mkt_acc:+.0%}) on {len(played)} graded games")

    st.subheader("Slate")
    # Sort the slate by the model's winning-side probability (most confident
    # games first, coin-flips last) — distinct from the edge-ranked cards above.
    confidence = df["model_home_prob"].apply(lambda p: max(p, 1 - p))
    by_prob = df.reindex(confidence.sort_values(ascending=False).index)
    display = _display_table(by_prob, graded)
    st.dataframe(display, width="stretch", hide_index=True)

    pdf_bytes = _preview_pdf(season, week, kind, display, disagreements, acc_line)
    st.download_button(
        "⬇ Download preview as PDF", data=pdf_bytes,
        file_name=f"nfl-preview-{season}-wk{week:02d}-{kind}.pdf",
        mime="application/pdf", key="download_preview_pdf")

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


def render_roster(team: str, season: int, starters_only: bool) -> None:
    with st.spinner(f"Loading {team} {season} roster…"):
        try:
            r = _roster(team, season)
        except SystemExit as e:
            st.error(str(e) or f"No roster data for {team} {season}.")
            return

    rated = r[r["overallrating"].notna()]
    starters = r[r["starter"]]

    # Headline: starter talent by unit (the same signals the model's features use).
    st.subheader(f"{team} {season} — starter talent")
    cols = st.columns(4)
    qb = starters[starters["position"] == "QB"]["overallrating"]
    units = [
        ("QB OVR", qb),
        ("Offense starters", starters[starters["unit"] == "Offense"]["overallrating"]),
        ("Defense starters", starters[starters["unit"] == "Defense"]["overallrating"]),
        ("All starters", starters["overallrating"]),
    ]
    for col, (label, series) in zip(cols, units):
        val = series.dropna()
        col.metric(label, f"{val.mean():.0f}" if len(val) else "—",
                   f"{len(val)} rated" if label != "QB OVR" else None,
                   delta_color="off")

    # Ratings distribution by unit.
    if not rated.empty:
        st.subheader("Player ratings (overall) by unit")
        chart = (
            alt.Chart(rated).mark_circle(size=90, opacity=0.7).encode(
                x=alt.X("overallrating:Q", title="Madden overall",
                        scale=alt.Scale(zero=False)),
                y=alt.Y("unit:N", title=None,
                        sort=["Offense", "Defense", "Special teams", "Other"]),
                color=alt.Color("starter:N", title="Starter",
                                scale=alt.Scale(domain=[True, False],
                                                range=["#1f77b4", "#cccccc"])),
                tooltip=["player", "position", "overallrating",
                         alt.Tooltip("snap_share:Q", title="Snap %", format=".0%")],
            ).properties(height=200)
        )
        st.altair_chart(chart, width="stretch")

    # Roster table.
    st.subheader("Roster" + (" — starters" if starters_only else ""))
    table = (starters if starters_only else r).copy()
    table["Snap %"] = (table["snap_share"] * 100).round(0).astype("Int64")
    table["Starter"] = table["starter"].map({True: "✓", False: ""})
    show = table.rename(columns={
        "player": "Player", "position": "Pos", "unit": "Unit",
        "games": "GP", "overallrating": "OVR", "speed": "SPD",
        "acceleration": "ACC", "awareness": "AWR", "strength": "STR",
    })
    order = [c for c in ["Player", "Pos", "Unit", "OVR", "Snap %", "Starter",
                         "GP", "SPD", "ACC", "AWR", "STR"] if c in show.columns]
    st.dataframe(show[order], width="stretch", hide_index=True)
    st.caption("Starters = season-average snap share ≥ 50% on offense or defense. "
               "Ratings are Madden launch OVR joined by player id; blanks = unrated.")


def render_pickem(season: int, through_week: int, train_start: int, kind: str) -> None:
    with st.spinner(f"Grading {season} and scoring picks…"):
        try:
            s = _grade(season, through_week, train_start, kind)
        except SystemExit as e:
            st.error(str(e) or f"Nothing to grade for {season}.")
            return

    all_picks = picks_mod.load_all_picks(season, through_week)
    if all_picks.empty:
        st.info(
            f"No picks found for {season} (through Week {through_week}). Seed a "
            "week with `uv run picks.py --season {0} --week N`, fill in the "
            "`pick`/`confidence` columns, then re-run.".format(season))
        return

    scored = picks_mod.score(all_picks, s)
    if scored.empty:
        st.warning("Picks were found but none matched a completed game yet.")
        return

    board = picks_mod.leaderboard(scored, s)

    st.subheader(f"Standings — through Week {int(s['week'].max())}")
    leader = board.iloc[0]
    cols = st.columns(min(len(board), 4))
    for col, (_, r) in zip(cols, board.iterrows()):
        col.metric(r["Player"], r["Record"], f"vs model {r['vs Model']}",
                   delta_color="off")
    st.caption(f"🏆 Leading: **{leader['Player']}** ({leader['Record']}). "
               "“vs Model” is each player’s accuracy minus the model’s over the "
               "same games they picked.")

    st.dataframe(board, width="stretch", hide_index=True)

    # Accuracy bar — who's beating the model baseline.
    chart_df = board.copy()
    chart_df["AccPct"] = scored.groupby("player")["correct"].mean().reindex(
        chart_df["Player"]).to_numpy() * 100
    bar = (
        alt.Chart(chart_df).mark_bar().encode(
            x=alt.X("AccPct:Q", title="Straight-up accuracy (%)"),
            y=alt.Y("Player:N", sort="-x", title=None),
            color=alt.Color("AccPct:Q", scale=alt.Scale(scheme="greens"),
                            legend=None),
            tooltip=["Player", "Record", "vs Model", "Brier", "Log loss"],
        ).properties(height=max(140, 34 * len(chart_df)))
    )
    st.altair_chart(bar, width="stretch")

    # This week's head-to-head.
    this_week = scored[scored["week"] == int(s["week"].max())]
    if not this_week.empty:
        st.subheader(f"Week {int(s['week'].max())} — game by game")
        wk = this_week.assign(
            Matchup=this_week["away_team"] + " @ " + this_week["home_team"],
            Result=this_week.apply(
                lambda r: f"{r['pick']} {'✓' if r['correct'] else '✗'}", axis=1),
        )
        pivot = wk.pivot_table(index=["Matchup"], columns="player",
                               values="Result", aggfunc="first").reset_index()
        st.dataframe(pivot, width="stretch", hide_index=True)

    st.caption("Picks come from predictions/picks/*.csv (one row per game/player). "
               "Brier / log loss use only picks that carried a confidence.")


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

preview_tab, tracker_tab, pickem_tab, roster_tab = st.tabs(
    ["Weekly preview", "Season tracker", "Pick'em leaderboard", "Team roster"])

with preview_tab:
    st.title(f"Preview — {season}")
    week = st.number_input("Week", min_value=1, max_value=22, value=1, step=1)
    if st.button("Run preview", type="primary", key="run_preview"):
        st.session_state["preview"] = (int(season), int(week), int(train_start), kind)
    # Persist the last-run preview so the rerun triggered by the PDF download
    # button (or any widget) re-renders it instead of blanking the tab.
    if "preview" in st.session_state:
        render_preview(*st.session_state["preview"])
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

with pickem_tab:
    st.title(f"Pick'em leaderboard — {season}")
    through_pk = st.number_input("Through week", min_value=1, max_value=22, value=1,
                                 step=1, key="pickem_week")
    if st.button("Run leaderboard", type="primary", key="run_pickem"):
        render_pickem(int(season), int(through_pk), int(train_start), kind)
    else:
        st.info("Tracks you and your friends vs the model. Seed a week with "
                "`uv run picks.py --season {0} --week N`, fill in everyone's "
                "`pick`/`confidence`, then **Run leaderboard**.".format(int(season)))

with roster_tab:
    st.title(f"Team roster — {season}")
    c1, c2 = st.columns([2, 1])
    team = c1.selectbox("Team", _teams(int(season)), key="roster_team")
    starters_only = c2.checkbox("Starters only", value=True, key="roster_starters")
    if st.button("Show roster", type="primary", key="run_roster"):
        render_roster(team, int(season), starters_only)
    else:
        st.info("Pick a team and **Show roster** for its starters and Madden "
                "player ratings. Needs a season that's already underway.")
