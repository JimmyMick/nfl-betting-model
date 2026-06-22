"""Cloud (read-only) dashboard for the NFL model — Streamlit Community Cloud.

Renders the artifacts exported by the local weekly runs (see
``nfl_betting_model/cloud.py``): the pick'em leaderboard, the season tracker, and
the latest weekly preview. It does **no** training and never fetches data, so it
runs comfortably in the free tier's ~1 GB. The full, live-training app is
``dashboard.py`` (run locally).

Deploy: point Streamlit Community Cloud at this repo and this file
(``streamlit_app.py``). Dependencies come from ``requirements.txt`` (the light
set — pandas / numpy / sklearn / altair / streamlit).
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import brier_score_loss, log_loss

from nfl_betting_model import cloud, picks as picks_mod

st.set_page_config(page_title="NFL model — leaderboard", page_icon="🏈",
                   layout="wide")


# ── Optional Descope (OIDC) sign-in gate ──────────────────────────────────────
# Uses Streamlit's native OIDC login. Entirely inert until an [auth] block is
# configured in the app's secrets, so the app keeps working open before setup.
# Configure a provider named [auth.descope] and (optionally) an
# [access] allowed_emails list to restrict who gets in.
def _auth_configured() -> bool:
    try:
        return "auth" in st.secrets
    except Exception:
        return False


def _require_login() -> None:
    if not _auth_configured():
        return  # open mode — no auth secrets configured yet
    if not st.user.is_logged_in:
        st.title("🏈 NFL model — pick'em & tracker")
        st.write("This leaderboard is private. Sign in to continue.")
        st.button("Log in with Descope", type="primary",
                  on_click=st.login, args=["descope"])
        st.stop()

    email = getattr(st.user, "email", None)
    try:
        allowed = list(st.secrets.get("access", {}).get("allowed_emails", []))
    except Exception:
        allowed = []
    if allowed and email not in allowed:
        st.error(f"{email or 'This account'} isn't on the access list for this app.")
        st.button("Log out", on_click=st.logout)
        st.stop()

    with st.sidebar:
        st.caption(f"Signed in as {getattr(st.user, 'name', None) or email}")
        st.button("Log out", on_click=st.logout)


# ── Small grade helpers (reimplemented here to keep cloud imports light — the
#    originals live in grade.py, which pulls in the heavy training stack). ──────
def _record(correct: pd.Series) -> str:
    w = int(correct.sum())
    n = len(correct)
    return f"{w}-{n - w} ({w / n:.0%})" if n else "0-0 (—)"


def _prob_str(home_team: str, away_team: str, home_prob: float) -> str:
    if home_prob >= 0.5:
        return f"{home_team} {home_prob:.0%}"
    return f"{away_team} {1 - home_prob:.0%}"


def _calibration(g: pd.DataFrame) -> tuple[float, float, float, float]:
    y = g["home_win"].to_numpy()
    pm = g["model_home_prob"].to_numpy()
    m_ll = log_loss(y, pm, labels=[0, 1])
    m_br = brier_score_loss(y, pm)
    mkt = g["market_home_prob"].to_numpy()
    mask = ~np.isnan(mkt)
    if mask.sum():
        k_ll = log_loss(y[mask], mkt[mask], labels=[0, 1])
        k_br = brier_score_loss(y[mask], mkt[mask])
    else:
        k_ll = k_br = float("nan")
    return m_ll, m_br, k_ll, k_br


def _top_picks(g: pd.DataFrame) -> pd.DataFrame:
    """The model's most-confident pick in each week, with the actual result."""
    conf = g["model_home_prob"].apply(lambda p: max(p, 1 - p))
    g = g.assign(_conf=conf)
    rows = []
    for wk, grp in g.groupby("week"):
        r = grp.loc[grp["_conf"].idxmax()]
        rows.append({
            "Week": str(int(wk)),
            "Matchup": f"{r['away_team']} @ {r['home_team']}",
            "Top pick": r["model_pick"],
            "Confidence": f"{r['_conf']:.0%}",
            "Actual": r["winner"],
            "Result": "✓" if r["model_correct"] else "✗",
        })
    return pd.DataFrame(rows)


def _topn_correct(g: pd.DataFrame, n: int) -> pd.Series:
    """model_correct for the n most-confident games in each week."""
    conf = g["model_home_prob"].apply(lambda p: max(p, 1 - p))
    g = g.assign(_conf=conf)
    parts = [grp.sort_values("_conf", ascending=False).head(n)
             for _, grp in g.groupby("week")]
    return pd.concat(parts)["model_correct"]


def _weekly_summary(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for wk, grp in g.groupby("week"):
        rows.append({"Week": str(int(wk)), "Games": len(grp),
                     "Model": _record(grp["model_correct"]),
                     "Market": _record(grp["market_correct"])})
    out = pd.DataFrame(rows)
    out.loc[len(out)] = {"Week": "Season", "Games": len(g),
                         "Model": _record(g["model_correct"]),
                         "Market": _record(g["market_correct"])}
    return out


# ── Tab renderers ─────────────────────────────────────────────────────────────
def render_leaderboard(scored: pd.DataFrame | None, graded: pd.DataFrame) -> None:
    if scored is None or scored.empty:
        st.info("No picks recorded yet. Once players submit picks and a week is "
                "graded, the leaderboard populates here — each player scored "
                "against the model on the games they picked.")
        return

    board = picks_mod.leaderboard(scored, graded)
    leader = board.iloc[0]

    st.subheader("Standings")
    cols = st.columns(min(len(board), 5))
    for col, (_, r) in zip(cols, board.iterrows()):
        col.metric(r["Player"], r["Record"], f"vs model {r['vs Model']}",
                   delta_color="off")
    st.caption(f"🏆 Leading: **{leader['Player']}** ({leader['Record']}). "
               "“vs Model” = a player’s accuracy minus the model’s over the same "
               "games they picked.")
    st.dataframe(board, width="stretch", hide_index=True)

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

    last_week = int(scored["week"].max())
    this_week = scored[scored["week"] == last_week]
    if not this_week.empty:
        st.subheader(f"Week {last_week} — game by game")
        wk = this_week.assign(
            Matchup=this_week["away_team"] + " @ " + this_week["home_team"],
            Result=this_week.apply(
                lambda r: f"{r['pick']} {'✓' if r['correct'] else '✗'}", axis=1))
        pivot = wk.pivot_table(index=["Matchup"], columns="player",
                               values="Result", aggfunc="first").reset_index()
        st.dataframe(pivot, width="stretch", hide_index=True)
    st.caption("Brier / log loss use only picks that carried a confidence.")


def render_tracker(graded: pd.DataFrame) -> None:
    m_acc = graded["model_correct"].mean()
    k_acc = graded["market_correct"].mean()
    m_ll, m_br, k_ll, k_br = _calibration(graded)

    st.subheader(f"Season-to-date — through Week {int(graded['week'].max())}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Model straight-up", _record(graded["model_correct"]))
    c2.metric("Market straight-up", _record(graded["market_correct"]))
    c3.metric("vs market", f"{m_acc - k_acc:+.0%}")
    c4, c5 = st.columns(2)
    c4.metric("Model calibration", f"logloss {m_ll:.3f}", f"Brier {m_br:.3f}",
              delta_color="off")
    c5.metric("Market calibration", f"logloss {k_ll:.3f}", f"Brier {k_br:.3f}",
              delta_color="off")

    st.subheader("Accuracy ticker (cumulative)")
    wk = graded.sort_values(["week"]).copy()
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

    st.subheader("Top pick of the week (most confident)")
    top = _top_picks(graded)
    t1, t3 = st.columns(2)
    t1.metric("Top pick record", _record(top["Result"] == "✓"))
    t3.metric("Top-3 picks record", _record(_topn_correct(graded, 3)))
    st.dataframe(top, width="stretch", hide_index=True)
    st.caption("Each week's single highest-confidence model pick vs. the actual "
               "result — the model's “lock of the week.” The Top-3 record pools "
               "the three most-confident games each week.")

    st.subheader("Week-by-week")
    st.dataframe(_weekly_summary(graded), width="stretch", hide_index=True)
    st.caption("Scorekeeping companion to the preview. Market-grade calibration "
               "is expected — the model is a forecaster, not a beater.")


def render_preview(preview: pd.DataFrame) -> None:
    df = preview.copy()
    df["fav"] = np.where(df["edge"] > 0, df["home_team"], df["away_team"])
    by_edge = df.reindex(df["edge"].abs().sort_values(ascending=False).index)

    st.subheader("Biggest model-vs-market disagreements")
    for col, (_, r) in zip(st.columns(3), by_edge.head(3).iterrows()):
        col.metric(f"{r['away_team']} @ {r['home_team']}",
                   f"{r['fav']} +{abs(r['edge']):.0%}", r["driver"],
                   delta_color="off")

    st.subheader("Slate")
    conf = df["model_home_prob"].apply(lambda p: max(p, 1 - p))
    by_prob = df.reindex(conf.sort_values(ascending=False).index)
    rows = []
    for _, r in by_prob.iterrows():
        rows.append({
            "Matchup": f"{r['away_team']} @ {r['home_team']}",
            "Model": _prob_str(r["home_team"], r["away_team"], r["model_home_prob"]),
            "Market": _prob_str(r["home_team"], r["away_team"], r["market_home_prob"]),
            "Edge": f"{r['fav']} +{abs(r['edge']):.0%}",
            "Key driver": r["driver"],
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption("Probability/preview tool — model edges are disagreement with the "
               "closing line, not a betting signal (moneyline is efficient).")


# ── Page ──────────────────────────────────────────────────────────────────────
_require_login()

st.title("🏈 NFL model — pick'em & tracker")

art = cloud.load_artifacts()
graded, scored, preview, meta = (
    art["graded"], art["scored"], art["preview"], art["meta"])

if graded is None and preview is None:
    st.warning("No data published yet. The local weekly runs export results here "
               "(`predictions/cloud/`) and push them; this app renders whatever's "
               "been published.")
    st.stop()

# Freshness line.
stamps = []
if meta.get("grade_generated_at"):
    stamps.append(f"grade through Wk {meta.get('grade_through_week', '?')} "
                  f"({meta['grade_generated_at'][:10]})")
if meta.get("preview_generated_at"):
    stamps.append(f"preview Wk {meta.get('preview_week', '?')} "
                  f"({meta['preview_generated_at'][:10]})")
season = meta.get("grade_season") or meta.get("preview_season") or ""
if stamps:
    st.caption(f"**{season} season** · last updated: " + " · ".join(stamps))

tabs, names = [], []
if scored is not None or graded is not None:
    names.append("Pick'em leaderboard")
if graded is not None:
    names.append("Season tracker")
if preview is not None:
    names.append("Weekly preview")
made = st.tabs(names)
tab_by_name = dict(zip(names, made))

if "Pick'em leaderboard" in tab_by_name:
    with tab_by_name["Pick'em leaderboard"]:
        if graded is None:
            st.info("Leaderboard needs a graded week to score against.")
        else:
            render_leaderboard(scored, graded)

if "Season tracker" in tab_by_name:
    with tab_by_name["Season tracker"]:
        render_tracker(graded)

if "Weekly preview" in tab_by_name:
    with tab_by_name["Weekly preview"]:
        render_preview(preview)
