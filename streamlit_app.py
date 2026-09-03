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

import datetime as dt
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import brier_score_loss, log_loss

from nfl_betting_model import (
    cloud, paper as paper_mod, picks as picks_mod, submit as submit_mod,
    teams as teams_mod)

st.set_page_config(page_title="NFL model — leaderboard", page_icon="🏈",
                   layout="wide")


# ── Team-logo helpers (small helmet icons next to abbreviations) ───────────────
def _logo_col(label: str = "") -> "st.column_config.ImageColumn":
    return st.column_config.ImageColumn(label, width="small")


def team_logos(abbrs) -> pd.Series:
    """Map a series/list of team abbreviations to their logo URLs."""
    return pd.Series(abbrs).map(teams_mod.logo).values


def matchup_frame(away, home) -> dict:
    """Columns for an ``away @ home`` row: away logo/abbr then home logo/abbr."""
    return {"": team_logos(away), "Away": list(away),
            " ": team_logos(home), "Home": list(home)}


def logo_cfg(*keys) -> dict:
    """column_config mapping each (whitespace) key to a small logo image column."""
    return {k: _logo_col() for k in keys}


MATCHUP_CFG = logo_cfg("", " ")


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
            **matchup_frame([r["away_team"]], [r["home_team"]]),
            "  ": teams_mod.logo(r["model_pick"]), "Top pick": r["model_pick"],
            "Confidence": f"{r['_conf']:.0%}",
            "   ": teams_mod.logo(r["winner"]) if pd.notna(r["winner"]) else None,
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
    st.caption("👋 New here? The **📖 Guide** tab covers how to enter your picks "
               "and read everything below.")
    if scored is None or scored.empty:
        st.info("No picks recorded yet. Once the Urban Platform Experts submit picks and a week is "
                "graded, the leaderboard populates here — each expert scored "
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
               "“vs Model” = an expert’s accuracy minus the model’s over the same "
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

        # AI expert reasoning (any pick that carried a written rationale).
        if "rationale" in this_week.columns:
            rat = this_week[this_week["rationale"].notna()
                            & this_week["rationale"].astype(str).str.strip().ne("")]
            if not rat.empty:
                with st.expander("🤖 AI expert — why it picked what it did"):
                    rr = rat.assign(
                        Matchup=rat["away_team"] + " @ " + rat["home_team"],
                        Pick=rat.apply(
                            lambda r: f"{r['pick']} {'✓' if r['correct'] else '✗'}",
                            axis=1)).rename(columns={"player": "Player",
                                                     "rationale": "Rationale"})
                    st.dataframe(rr[["Player", "Matchup", "Pick", "Rationale"]],
                                 width="stretch", hide_index=True)
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
    st.dataframe(top, width="stretch", hide_index=True,
                 column_config=logo_cfg("", " ", "  ", "   "))
    st.caption("Each week's single highest-confidence model pick vs. the actual "
               "result — the model's “lock of the week.” The Top-3 record pools "
               "the three most-confident games each week.")

    st.subheader("Week-by-week")
    st.dataframe(_weekly_summary(graded), width="stretch", hide_index=True)
    st.caption("Scorekeeping companion to the preview. Market-grade calibration "
               "is expected — the model is a forecaster, not a beater.")


# A sub-0.5% edge rounds to 0% — the model and market agree and the sign is
# noise, so render a dash instead of a spurious "TEAM +0%". (Mirrors
# predict.edge_label; kept local so the cloud app avoids the heavy predict import.)
_EDGE_ZERO = 0.005


def _edge_label(edge: float, home_team: str, away_team: str) -> str:
    if pd.isna(edge) or abs(edge) < _EDGE_ZERO:
        return "—"
    side = home_team if edge > 0 else away_team
    return f"{side} +{abs(edge):.0%}"


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
    bp = df.reindex(conf.sort_values(ascending=False).index).copy()
    bp["Model"] = bp.apply(
        lambda r: _prob_str(r["home_team"], r["away_team"], r["model_home_prob"]), axis=1)
    bp["Market"] = bp.apply(
        lambda r: _prob_str(r["home_team"], r["away_team"], r["market_home_prob"]), axis=1)
    bp["Edge"] = bp.apply(
        lambda r: _edge_label(r["edge"], r["home_team"], r["away_team"]), axis=1)
    show = pd.DataFrame({
        **matchup_frame(bp["away_team"], bp["home_team"]),
        "Model": bp["Model"].values, "Market": bp["Market"].values,
        "Edge": bp["Edge"].values, "Key driver": bp["driver"].values,
    })
    st.dataframe(show, width="stretch", hide_index=True, column_config=MATCHUP_CFG)
    st.caption("**Edge** = the side the model values *more than the market prices "
               "it* — a mispricing, not a winner pick. It can name the underdog "
               "even when the model still expects the favourite to win (it just "
               "rates the favourite lower than Vegas). Probability/preview tool: "
               "edges are disagreement with the closing line, not a betting "
               "signal (moneyline is efficient).")


def _schedule_rows(sched: pd.DataFrame) -> pd.DataFrame:
    """Shape a raw schedule frame into a display table (Date / Matchup / Result)."""
    df = sched.copy()
    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    day = pd.to_datetime(df.get("gameday"), errors="coerce")
    df["Date"] = day.dt.strftime("%a %b %d")
    df["Matchup"] = df["away_team"].astype(str) + " @ " + df["home_team"].astype(str)

    aw = pd.to_numeric(df.get("away_score"), errors="coerce")
    hm = pd.to_numeric(df.get("home_score"), errors="coerce")

    def _result(r) -> str:
        a, h = aw.get(r.name), hm.get(r.name)
        if pd.isna(a) or pd.isna(h):
            return "—"
        winner = r["home_team"] if h > a else r["away_team"] if a > h else "Tie"
        return f"{r['away_team']} {int(a)}–{int(h)} {r['home_team']}  ✓ {winner}"

    df["Result"] = df.apply(_result, axis=1)
    played = aw.notna() & hm.notna()
    return df, played


def render_schedule(sched: pd.DataFrame | None, season) -> None:
    """Full-season matchup schedule, filterable by week (defaults to the next
    unplayed week)."""
    st.subheader(f"{season} schedule" if season else "Schedule")
    if sched is None or sched.empty:
        st.info("Schedule hasn't been published yet.")
        return

    df, played = _schedule_rows(sched)
    weeks = sorted(int(w) for w in df["week"].dropna().unique())
    upcoming = [w for w in weeks if not played[df["week"] == w].all()]
    default_week = upcoming[0] if upcoming else (weeks[-1] if weeks else 1)

    labels = ["All weeks"] + [f"Week {w}" for w in weeks]
    default_label = f"Week {default_week}" if default_week in weeks else "All weeks"
    choice = st.selectbox("Week", labels, index=labels.index(default_label))

    view = df if choice == "All weeks" else df[df["week"] == int(choice.split()[1])]
    show = pd.DataFrame({
        "Wk": view["week"].values, "Date": view["Date"].values,
        **matchup_frame(view["away_team"], view["home_team"]),
        "Result": view["Result"].values,
    })
    st.dataframe(show, width="stretch", hide_index=True, column_config=MATCHUP_CFG)
    n_played = int(played[view.index].sum())
    st.caption(f"{len(view)} games · {n_played} played · "
               "results fill in as the weekly runs refresh.")


def render_playoff_odds(sim, sim_history, meta, season) -> None:
    """Monte Carlo playoff-odds projection + week-to-week trend."""
    through = meta.get("sim_through_week")
    when = "preseason" if through in (0, None) else f"through Week {int(through)}"
    st.subheader(f"Playoff odds — {season} ({when})")
    if sim is None or sim.empty:
        st.info("No simulation published yet.")
        return

    df = sim.sort_values("win_sb", ascending=False).reset_index(drop=True)
    show = pd.DataFrame({
        "": team_logos(df["team"]),
        "Team": df["team"].values, "Conf": df["conference"].values,
        "Proj W": df["proj_wins"].map(lambda x: f"{x:.1f}").values,
        "Playoffs": df["make_playoffs"].map(lambda x: f"{x:.0%}").values,
        "Division": df["win_division"].map(lambda x: f"{x:.0%}").values,
        "#1 Seed": df["top_seed"].map(lambda x: f"{x:.0%}").values,
        "Conf ": df["win_conference"].map(lambda x: f"{x:.0%}").values,
        "Super Bowl": df["win_sb"].map(lambda x: f"{x:.0%}").values,
    })
    st.dataframe(show, width="stretch", hide_index=True,
                 column_config={"": _logo_col()})

    top = df.head(12)
    chart = alt.Chart(top).mark_bar().encode(
        x=alt.X("win_sb:Q", title="Super Bowl odds", axis=alt.Axis(format="%")),
        y=alt.Y("team:N", sort="-x", title=None),
        color=alt.Color("conference:N", legend=None),
        tooltip=["team", alt.Tooltip("win_sb:Q", format=".1%")],
    ).properties(height=320)
    st.altair_chart(chart, use_container_width=True)

    # Week-to-week trend once more than one snapshot exists.
    if sim_history is not None and not sim_history.empty:
        hist = sim_history[sim_history["season"] == season]
        if hist["through_week"].nunique() > 1:
            st.subheader("Week-to-week trend")
            team = st.selectbox("Team", sorted(hist["team"].unique()))
            th = hist[hist["team"] == team]
            melt = th.melt(id_vars="through_week",
                           value_vars=["make_playoffs", "win_sb"],
                           var_name="metric", value_name="prob")
            line = alt.Chart(melt).mark_line(point=True).encode(
                x=alt.X("through_week:Q", title="through week"),
                y=alt.Y("prob:Q", axis=alt.Axis(format="%")),
                color="metric:N",
            ).properties(height=260)
            st.altair_chart(line, use_container_width=True)

    st.caption("Monte Carlo of the rest of the season using the model. Ratings "
               "held fixed across each sim; playoff games use current Elo + "
               "home field. **Just for fun — not a betting product.**")


def render_paper(ledger: pd.DataFrame) -> None:
    """The out-of-sample paper-trade tracker for the single biggest disagreement.

    Each week the model backs its side of the largest model-vs-market gap, flat
    10 units at the previewed moneyline. A 2016-2025 backtest returned +23% here
    (bootstrap-clear of zero) but that's in-sample; this is the forward test.
    """
    st.subheader("Paper play — biggest disagreement of the week")
    st.caption("A flat **10u** paper bet on the model's side of the single "
               "largest model-vs-market disagreement each week. No real money — "
               "this is the honest **out-of-sample** test of a signal that "
               "looked strong (+23% ROI) in a 2016-2025 backtest.")

    if ledger is None or ledger.empty:
        st.info("No paper plays logged yet. The first one posts when a weekly "
                "preview runs — one play per week, graded the following Tuesday.")
        return

    s = paper_mod.summary(ledger)
    w = paper_mod.whatif_summary(ledger)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Record", f"{s['wins']}-{s['losses']}")
    c2.metric("Profit", f"{s['profit']:+.1f}u")
    c3.metric("ROI", f"{s['roi']:+.1%}" if s["bets"] else "—")
    c4.metric("Open", s["open"])
    if w["bets"]:
        st.caption(f"**What-if ({paper_mod.WHATIF_NAME}):** {w['profit']:+.1f}u on "
                   f"{w['staked']:.0f}u staked · ROI {w['roi']:+.1%} — a *derived* "
                   "alternative where the stake scales with the size of the edge "
                   "(capped at 3×). The flat 10u line stays the tracked baseline.")

    led = paper_mod.add_whatif(ledger.sort_values(["season", "week"]).copy())
    settled = led[led["result"].isin(["win", "loss", "push"])]
    if not settled.empty:
        settled = settled.assign(
            label=settled["season"].astype(int).astype(str)
            + " Wk" + settled["week"].astype(int).astype(str))
        settled["Flat 10u"] = settled["profit"].cumsum()
        settled[f"What-if ({paper_mod.WHATIF_NAME})"] = \
            settled["whatif_profit"].cumsum()
        long = settled.melt(
            id_vars=["label"],
            value_vars=["Flat 10u", f"What-if ({paper_mod.WHATIF_NAME})"],
            var_name="Curve", value_name="cum")
        line = (
            alt.Chart(long).mark_line(point=True).encode(
                x=alt.X("label:N", sort=None, title=None),
                y=alt.Y("cum:Q", title="Cumulative units"),
                color=alt.Color("Curve:N", title=None,
                                scale=alt.Scale(range=["#3987e5", "#199e70"])),
                tooltip=["label", "Curve",
                         alt.Tooltip("cum:Q", title="Units", format="+.1f")],
            ).properties(height=280)
        )
        st.altair_chart(line, width="stretch")

    rows = []
    for _, r in led.iterrows():
        if r["result"] == "win":
            res = f"✓ +{float(r['profit']):.1f}u"
        elif r["result"] == "loss":
            res = f"✗ {float(r['profit']):.1f}u"
        elif r["result"] == "open":
            res = "open"
        else:
            res = str(r["result"])
        price = f"{int(r['price_ml']):+d}" if pd.notna(r["price_ml"]) else "n/a"
        if pd.notna(r["whatif_profit"]):
            wi = f"{float(r['whatif_stake']):.0f}u → {float(r['whatif_profit']):+.1f}u"
        elif r["result"] == "open" and pd.notna(r["whatif_stake"]):
            wi = f"{float(r['whatif_stake']):.0f}u staked"
        else:
            wi = "—"
        rows.append({
            "Week": int(r["week"]),
            "": teams_mod.logo(r["model_side"]),
            "Play": f"{r['model_side']} ({r['away_team']} @ {r['home_team']})",
            "Edge": f"+{abs(float(r['edge'])):.0%}" if pd.notna(r["edge"]) else "—",
            "Price": price,
            "Result": res,
            "What-if": wi,
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True,
                 column_config=logo_cfg(""))
    st.caption("One bet per week, priced when the preview ran (Thursday). "
               "Backtest was strong but in-sample — trust the forward record. "
               "The **What-if** column is a derived edge-proportional stake, not "
               "a second real tracker.")


def render_guide() -> None:
    """Render the friend-facing user guide (GUIDE.md at the repo root)."""
    try:
        st.markdown((Path(__file__).parent / "GUIDE.md").read_text())
    except Exception:
        st.info("Guide not available.")


def render_open_ai_picks(meta: dict) -> None:
    """The AI expert's picks for the OPEN (ungraded) week, shown pre-game.

    Reads the committed pick sheet for the preview week and shows only rows that
    carry a rationale — i.e. the AI expert. Human picks (no rationale) stay
    hidden until the week is graded, so the pool isn't spoiled.
    """
    season, week = meta.get("preview_season"), meta.get("preview_week")
    if season is None or week is None:
        return
    try:
        path = picks_mod.week_path(int(season), int(week))
        if not path.exists():
            return
        df = pd.read_csv(path, dtype={"game_id": str})
    except Exception:
        return
    if "rationale" not in df.columns:
        return
    ai = df[df["rationale"].notna()
            & df["rationale"].astype(str).str.strip().ne("")]
    if ai.empty:
        return
    st.subheader(f"🤖 AI expert — Week {int(week)} picks (pre-game)")
    show = pd.DataFrame({
        "Expert": ai["player"].values,
        **matchup_frame(ai["away_team"], ai["home_team"]),
        "  ": team_logos(ai["pick"]), "Pick": ai["pick"].values,
        "Conf": ai["confidence"].values, "Why": ai["rationale"].values,
    })
    st.dataframe(show, width="stretch", hide_index=True,
                 column_config=logo_cfg("", " ", "  "))
    st.caption("The AI's reasoning, shared **before kickoff**. The human experts' "
               "picks stay hidden until the week is graded.")


def _player_name() -> str | None:
    """Map the signed-in user's email to a pick'em expert name via [players]."""
    email = getattr(st.user, "email", None)
    if not email:
        return None
    try:
        mapping = dict(st.secrets.get("players", {}))
    except Exception:
        mapping = {}
    return mapping.get(email)


def _github_store() -> "submit_mod.GitHubStore | None":
    try:
        gh = st.secrets.get("github", {})
        token, repo = gh.get("token"), gh.get("repo")
    except Exception:
        return None
    if not token or not repo:
        return None
    return submit_mod.GitHubStore(token=token, repo=repo,
                                  branch=gh.get("branch", "main"))


def render_make_picks(preview: pd.DataFrame, meta: dict) -> None:
    season = meta.get("preview_season")
    week = meta.get("preview_week")
    if preview is None or season is None or week is None:
        st.info("No open slate to pick yet — the weekly preview hasn't been "
                "published. Picks open once that week's preview is exported.")
        return

    # Off-season: the published slate is last season's finale (a seeded sample),
    # so don't show a stale, unsubmittable week. Compare the previewed season to
    # the current NFL season (Sep-Dec = this year, Jan-Feb = last year, else the
    # upcoming year). The preview flips to the new season automatically once the
    # Week 1 slate is exported (~5 days before kickoff).
    _now = dt.datetime.now()
    _cur_season = (_now.year if _now.month >= 9
                   else _now.year - 1 if _now.month <= 2 else _now.year)
    if int(season) < _cur_season:
        st.info(f"🏈 **Picks open when the {_cur_season} season kicks off** — the "
                f"Week 1 slate posts around **early September {_cur_season}**, and "
                "this tab will switch to it automatically. What you see elsewhere "
                "is last season's finale, kept as a sample until then.")
        return

    if not _auth_configured() or not getattr(st.user, "is_logged_in", False):
        st.info("Sign-in must be enabled to submit picks (we attribute each pick "
                "to the signed-in expert). The leaderboard still works read-only.")
        return

    store = _github_store()
    if store is None:
        st.info("Pick submission isn't enabled yet. Add a `[github]` token to the "
                "app's secrets to turn it on; until then, picks are entered via "
                "the CSV sheets in the repo.")
        return

    player = _player_name()
    if player is None:
        st.warning(f"{getattr(st.user, 'email', 'This account')} isn't mapped to an "
                   "expert. Add it under `[players]` in the app secrets.")
        return

    st.subheader(f"Your picks — {season} Week {int(week)}")
    st.caption(f"Signed in as **{player}**. Pick a winner per game and set your "
               "confidence (50 = coin-flip, 100 = lock). Submitting overwrites "
               "your previous picks for this week.")

    games = preview[["away_team", "home_team"]].reset_index(drop=True)

    # Prefill from this expert's already-committed picks, if any.
    prior: dict[str, dict] = {}
    try:
        path = f"predictions/picks/{season}-wk{int(week):02d}.csv"
        current, _ = store.get_file(path)
        if current:
            import io
            cur = pd.read_csv(io.StringIO(current), dtype={"game_id": str})
            cur = cur[cur["player"].astype(str).str.strip() == player]
            for _, r in cur.iterrows():
                prior[str(r["game_id"])] = {
                    "pick": str(r["pick"]) if pd.notna(r["pick"]) else "",
                    "confidence": r["confidence"]}
    except Exception as e:  # noqa: BLE001 — prefill is best-effort
        st.caption(f"(Couldn't load existing picks: {e})")

    with st.form("make_picks"):
        selections: dict[str, dict] = {}
        for _, g in games.iterrows():
            away, home = g["away_team"], g["home_team"]
            gid = submit_mod.game_id(season, int(week), away, home)
            pre = prior.get(gid, {})
            opts = [away, home]
            idx = opts.index(pre["pick"]) if pre.get("pick") in opts else 0
            c1, c2 = st.columns([2, 1])
            pick = c1.radio(f"{away} @ {home}", opts, index=idx,
                            horizontal=True, key=f"pick_{gid}")
            try:
                conf_default = int(float(pre.get("confidence")))
            except (TypeError, ValueError):
                conf_default = 50
            conf = c2.slider("confidence", 50, 100, conf_default,
                             key=f"conf_{gid}", label_visibility="collapsed")
            selections[gid] = {"pick": pick, "confidence": conf}
        submitted = st.form_submit_button("Submit my picks", type="primary")

    if submitted:
        try:
            submit_mod.submit_picks(store, int(season), int(week), games,
                                    player, selections)
            st.success(f"Saved {len(selections)} picks for {player}, "
                       f"Week {int(week)}. They'll score in Tuesday's grade run.")
        except Exception as e:  # noqa: BLE001
            st.error(f"Couldn't save picks: {e}")


# ── Page ──────────────────────────────────────────────────────────────────────
_require_login()

st.title("🏈 NFL model — pick'em & tracker")

art = cloud.load_artifacts()
graded, scored, preview, meta = (
    art["graded"], art["scored"], art["preview"], art["meta"])
schedule = art["schedule"]
sim = art["sim"]
sim_history = art["sim_history"]
paper_ledger = paper_mod.load_ledger()

if graded is None and preview is None and schedule is None and sim is None:
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
season = (meta.get("grade_season") or meta.get("preview_season")
          or meta.get("schedule_season") or meta.get("sim_season") or "")
if stamps:
    st.caption(f"**{season} season** · last updated: " + " · ".join(stamps))

tabs, names = [], []
if schedule is not None:
    names.append("🗓️ Schedule")
if sim is not None:
    names.append("🏆 Playoff odds")
if scored is not None or graded is not None:
    names.append("Pick'em leaderboard")
if preview is not None:
    names.append("Make picks")
if graded is not None:
    names.append("Season tracker")
if preview is not None:
    names.append("Weekly preview")
names.append("📈 Paper play")  # always shown; empty-state until the first play
names.append("📖 Guide")
made = st.tabs(names)
tab_by_name = dict(zip(names, made))

if "🗓️ Schedule" in tab_by_name:
    with tab_by_name["🗓️ Schedule"]:
        render_schedule(schedule, meta.get("schedule_season") or season)

if "🏆 Playoff odds" in tab_by_name:
    with tab_by_name["🏆 Playoff odds"]:
        render_playoff_odds(sim, sim_history, meta, meta.get("sim_season") or season)

if "Pick'em leaderboard" in tab_by_name:
    with tab_by_name["Pick'em leaderboard"]:
        if graded is None:
            st.info("Leaderboard needs a graded week to score against.")
        else:
            render_leaderboard(scored, graded)

if "Make picks" in tab_by_name:
    with tab_by_name["Make picks"]:
        render_make_picks(preview, meta)

if "Season tracker" in tab_by_name:
    with tab_by_name["Season tracker"]:
        render_tracker(graded)

if "Weekly preview" in tab_by_name:
    with tab_by_name["Weekly preview"]:
        render_preview(preview)
        render_open_ai_picks(meta)

if "📈 Paper play" in tab_by_name:
    with tab_by_name["📈 Paper play"]:
        render_paper(paper_ledger)

if "📖 Guide" in tab_by_name:
    with tab_by_name["📖 Guide"]:
        render_guide()
