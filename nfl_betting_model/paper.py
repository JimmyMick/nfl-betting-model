"""Paper-trade ledger for the single-biggest-disagreement play.

A backtest (2016-2025) found that flat-staking the model's side of the *single*
biggest model-vs-market disagreement each week returned +23% ROI, and the edge
survived a season-block bootstrap (95% CI clear of zero). That's promising but
in-sample: every modeling choice in this project was made on 2016-2025. The only
honest test is forward, on data the model was never tuned against.

So this module logs one paper play per week — the week's largest |edge| game,
backing the model's favoured side at the moneyline shown in the preview — and
settles it once results are final. No real money; this is the out-of-sample
scoreboard. A flat 10-unit stake, priced at the closing-ish line captured when
the preview ran.

Design notes
------------
* **One play per (season, week), first writer wins.** The Thursday early-look
  preview logs the play; the Saturday preview re-run is a no-op for an already
  logged week. That fixes a single, defensible bet-placement time.
* **Ledger lives under predictions/cloud/** so the existing weekly cron push
  (which already commits predictions/cloud) persists it and the read-only cloud
  app can render it. It is the record, not scratch output.
* Dependency-light (pandas + stdlib + betting.american_to_payout) to keep the
  cloud requirements tiny.
* **Logistic only, by design.** The play is logged from the live preview, which
  runs the logistic model — the one whose top-1 result cleared the bootstrap and
  whose tails are saner (top-1 is a tail event, exactly where gbm is weakest). If
  gbm's biggest-edge game differs, that does NOT change the tracked bet: requiring
  model agreement is a different, unbacktested strategy (fewer bets, needs its own
  bootstrap), and betting both is just top-2, which dilutes to break-even.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_betting_model.betting import american_to_payout
from nfl_betting_model.cloud import ARTIFACT_DIR

DEFAULT_STAKE = 10.0
PAPER_FILE = "paper_plays.csv"
LEDGER_PATH = ARTIFACT_DIR / PAPER_FILE

# One row per weekly play. ``result`` is one of open / win / loss / push /
# no_price; ``profit`` is in units and only meaningful once settled.
PLAY_COLS = [
    "season", "week", "game_id", "away_team", "home_team", "model_side",
    "bet_home", "model_home_prob", "market_home_prob", "edge", "price_ml",
    "stake", "result", "profit", "logged_at", "settled_at",
]

# Text columns are forced to object dtype on load: an all-NaN column otherwise
# reads back as float64 and rejects a later string write (e.g. settled_at).
_TEXT_COLS = ["game_id", "away_team", "home_team", "model_side", "result",
             "logged_at", "settled_at"]


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def load_ledger(path: Path = LEDGER_PATH) -> pd.DataFrame:
    """Read the ledger, returning an empty typed frame if it doesn't exist."""
    if Path(path).exists():
        df = pd.read_csv(path, dtype={"game_id": str})
        for c in PLAY_COLS:
            if c not in df.columns:
                df[c] = np.nan
        df = df[PLAY_COLS]
        for c in _TEXT_COLS:
            df[c] = df[c].astype(object)
        return df
    return pd.DataFrame({c: pd.Series(dtype=object if c in _TEXT_COLS else "float64")
                         for c in PLAY_COLS})


def _save(ledger: pd.DataFrame, path: Path = LEDGER_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ledger[PLAY_COLS].sort_values(["season", "week"]).to_csv(path, index=False)


def log_week(target: pd.DataFrame, season: int, week: int,
             stake: float = DEFAULT_STAKE, path: Path = LEDGER_PATH) -> dict | None:
    """Log the week's single biggest disagreement as a paper play.

    ``target`` is a predict.predict_week frame (already sorted by |edge| desc,
    but we re-sort defensively). Idempotent per (season, week): if the week is
    already logged, returns the existing row unchanged. Returns the play dict, or
    None if the slate is empty.
    """
    if target is None or target.empty:
        return None

    ledger = load_ledger(path)
    existing = ledger[(ledger["season"] == season) & (ledger["week"] == week)]
    if not existing.empty:
        return existing.iloc[0].to_dict()

    top = target.assign(_abs=target["edge"].abs()).sort_values(
        "_abs", ascending=False).iloc[0]
    bet_home = bool(top["edge"] > 0)
    model_side = top["home_team"] if bet_home else top["away_team"]
    price_ml = top.get("home_moneyline") if bet_home else top.get("away_moneyline")
    price_ml = float(price_ml) if pd.notna(price_ml) else np.nan

    play = {
        "season": int(season), "week": int(week), "game_id": str(top["game_id"]),
        "away_team": top["away_team"], "home_team": top["home_team"],
        "model_side": model_side, "bet_home": bet_home,
        "model_home_prob": round(float(top["model_home_prob"]), 4),
        "market_home_prob": round(float(top["market_home_prob"]), 4),
        "edge": round(float(top["edge"]), 4),
        "price_ml": price_ml, "stake": float(stake),
        "result": "no_price" if np.isnan(price_ml) else "open",
        "profit": np.nan, "logged_at": _now(), "settled_at": np.nan,
    }
    ledger = pd.concat([ledger, pd.DataFrame([play])], ignore_index=True)
    _save(ledger, path)
    return play


def settle(graded: pd.DataFrame, season: int,
           path: Path = LEDGER_PATH) -> pd.DataFrame:
    """Settle any open plays whose games are now final.

    ``graded`` is a grade.grade_season frame (game_id, home_win, winner). Matches
    open plays by game_id, computes win/loss + profit at the logged price, and
    writes the ledger back. Returns the updated ledger.
    """
    ledger = load_ledger(path)
    if ledger.empty:
        return ledger

    res = graded.set_index("game_id")["home_win"].to_dict()
    open_mask = (ledger["season"] == season) & (ledger["result"] == "open")
    for i in ledger[open_mask].index:
        gid = ledger.at[i, "game_id"]
        if gid not in res or pd.isna(res[gid]):
            continue
        home_win = int(res[gid])
        bet_home = bool(ledger.at[i, "bet_home"])
        won = (home_win == 1) if bet_home else (home_win == 0)
        stake = float(ledger.at[i, "stake"])
        payout = american_to_payout(pd.Series([ledger.at[i, "price_ml"]]))[0]
        ledger.at[i, "profit"] = stake * payout if won else -stake
        ledger.at[i, "result"] = "win" if won else "loss"
        ledger.at[i, "settled_at"] = _now()
    _save(ledger, path)
    return ledger


# ── "What-if" alternative stake curve ────────────────────────────────────────
# The live ledger stakes a flat 10u. This is a *derived* second column — an
# out-of-sample look at an edge-proportional curve (stake grows with the size of
# the disagreement), the highest-ROI scheme in the 2019-25 backtest. It is
# computed on the fly from the same bets/prices, so it never touches the stored
# ledger, the settle path, or the cron: the flat play stays the honest baseline.
WHATIF_NAME = "Edge-proportional"
WHATIF_REF_EDGE = 0.15   # an edge this big stakes exactly the base unit (10u)
WHATIF_MAX_MULT = 3.0    # cap so one monster edge can't stake more than 3x base


def whatif_stake(edge: float, base: float = DEFAULT_STAKE) -> float:
    """Edge-proportional stake: base * (|edge| / REF), capped at MAX_MULT."""
    if pd.isna(edge):
        return float("nan")
    mult = min(abs(float(edge)) / WHATIF_REF_EDGE, WHATIF_MAX_MULT)
    return base * mult


def add_whatif(ledger: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with derived ``whatif_stake`` / ``whatif_profit`` columns."""
    d = ledger.copy()
    d["whatif_stake"] = d["edge"].apply(whatif_stake)
    payout = american_to_payout(d["price_ml"])          # NaN-safe for no_price
    won = (d["result"] == "win").to_numpy()
    lost = (d["result"] == "loss").to_numpy()
    d["whatif_profit"] = np.where(
        won, d["whatif_stake"].to_numpy() * payout,
        np.where(lost, -d["whatif_stake"].to_numpy(), np.nan))
    return d


def whatif_summary(ledger: pd.DataFrame | None = None,
                   path: Path = LEDGER_PATH) -> dict:
    """Running record + ROI for the edge-proportional what-if curve."""
    if ledger is None:
        ledger = load_ledger(path)
    d = add_whatif(ledger)
    settled = d[d["result"].isin(["win", "loss", "push"])]
    staked = float(settled["whatif_stake"].sum())
    profit = float(settled["whatif_profit"].sum()) if len(settled) else 0.0
    return {"bets": len(settled), "staked": staked, "profit": profit,
            "roi": profit / staked if staked else float("nan")}


def summary(ledger: pd.DataFrame | None = None,
            path: Path = LEDGER_PATH) -> dict:
    """Running record + ROI over settled plays."""
    if ledger is None:
        ledger = load_ledger(path)
    settled = ledger[ledger["result"].isin(["win", "loss", "push"])]
    wins = int((settled["result"] == "win").sum())
    losses = int((settled["result"] == "loss").sum())
    staked = float(settled["stake"].sum())
    profit = float(settled["profit"].sum()) if len(settled) else 0.0
    return {
        "bets": len(settled), "wins": wins, "losses": losses,
        "open": int((ledger["result"] == "open").sum()),
        "staked": staked, "profit": profit,
        "roi": profit / staked if staked else float("nan"),
    }


def render(season: int, week: int, path: Path = LEDGER_PATH) -> list[str]:
    """Markdown block for the grade report: this week's play + running ledger."""
    ledger = load_ledger(path)
    if ledger.empty:
        return []
    lines = ["", "## 📝 Paper play — biggest disagreement (out-of-sample tracker)", ""]
    s = summary(ledger)
    roi = f"{s['roi']:+.1%}" if s["bets"] else "—"
    lines.append(
        f"**{s['wins']}-{s['losses']} settled · {s['profit']:+.1f}u on "
        f"{s['staked']:.0f}u staked · ROI {roi}** "
        f"({s['open']} open)  ·  flat {DEFAULT_STAKE:.0f}u, model's side of the "
        f"week's single largest model-vs-market gap"
    )
    w = whatif_summary(ledger)
    if w["bets"]:
        lines.append(
            f"_What-if ({WHATIF_NAME}): {w['profit']:+.1f}u on "
            f"{w['staked']:.0f}u staked · ROI {w['roi']:+.1%} — derived, "
            f"stake scales with edge size; the flat line above stays the "
            f"tracked baseline._")
    lines.append("")

    show = add_whatif(ledger[ledger["season"] == season].copy())
    if show.empty:
        return lines
    lines.append("| Week | Play | Edge | Price | Result | What-if |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in show.sort_values("week").iterrows():
        matchup = f"{r['away_team']} @ {r['home_team']}"
        play = f"{r['model_side']} ({matchup})"
        edge = f"+{abs(float(r['edge'])):.0%}" if pd.notna(r["edge"]) else "—"
        price = (f"{int(r['price_ml']):+d}" if pd.notna(r["price_ml"]) else "n/a")
        if r["result"] == "win":
            res = f"✓ +{float(r['profit']):.1f}u"
        elif r["result"] == "loss":
            res = f"✗ {float(r['profit']):.1f}u"
        elif r["result"] == "open":
            res = "open"
        else:
            res = str(r["result"])
        if pd.notna(r["whatif_profit"]):
            wi = (f"{float(r['whatif_stake']):.0f}u → "
                  f"{float(r['whatif_profit']):+.1f}u")
        elif r["result"] == "open" and pd.notna(r["whatif_stake"]):
            wi = f"{float(r['whatif_stake']):.0f}u staked"
        else:
            wi = "—"
        lines.append(
            f"| {int(r['week'])} | {play} | {edge} | {price} | {res} | {wi} |")
    return lines
