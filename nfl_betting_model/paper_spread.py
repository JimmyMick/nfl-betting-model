"""Paper-trade ledger for the spread (ATS) strategy — the second forward tracker.

Companion to ``paper.py`` (top-1 moneyline). A key-number-aware backtest (§7f,
`backtest_spread.py`) found that betting the model's side ATS when its implied
spread diverges from the market line by >= 2 points returns ~+2% ROI at ~53.4%
cover — the closest any strategy has come to an edge besides top-1, but it does
NOT clear the season-block bootstrap at 7 seasons (P(ROI>0) ~85%). So it is
tracked forward (not trusted) to accumulate out-of-sample seasons toward — or
away from — significance.

Each week: fit the empirical spread->P(home win) isotonic curve on all completed
games BEFORE this week (leak-free), invert it to map the model's win prob to an
implied home spread, and log a flat 10u -110 ATS bet on the model's side for
every game where |implied - spread_line| >= N (N=2, pre-registered). Settled vs
the actual margin once results are final. No real money.

Ledger lives under predictions/cloud/ so the existing weekly cron push persists
it and the read-only cloud app can render it. Dependency-light (pandas + sklearn
isotonic, already a project dep).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from nfl_betting_model import data
from nfl_betting_model.cloud import ARTIFACT_DIR

DEFAULT_N = 2.0                      # pre-registered spread-gap threshold (§7f)
STAKE = 10.0
VIG_PAYOUT = 100.0 / 110.0          # -110 both sides
FIRST_SNAP_SEASON = 2010
PAPER_FILE = "paper_spread_plays.csv"
LEDGER_PATH = ARTIFACT_DIR / PAPER_FILE
_GRID = np.arange(-21.0, 21.0001, 0.5)   # half-point spreads for the inversion

PLAY_COLS = [
    "season", "week", "game_id", "away_team", "home_team", "side", "bet_home",
    "model_home_prob", "spread_line", "implied_spread", "gap", "result",
    "profit", "logged_at", "settled_at",
]
_TEXT_COLS = ["game_id", "away_team", "home_team", "side", "result",
              "logged_at", "settled_at"]


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def load_ledger(path: Path = LEDGER_PATH) -> pd.DataFrame:
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
    ledger[PLAY_COLS].sort_values(["season", "week", "game_id"]).to_csv(
        path, index=False)


def _prob_grid(season: int, week: int) -> np.ndarray:
    """Empirical P(home win) at each grid spread, fit on games strictly before
    (season, week) — the same key-number-aware mapping as backtest_spread.py."""
    g = data.load_games(list(range(FIRST_SNAP_SEASON, season + 1)))
    g = g[g["home_win"].notna() & g["spread_line"].notna()]
    g = g[(g["season"] < season) | ((g["season"] == season) & (g["week"] < week))]
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(g["spread_line"].to_numpy(), g["home_win"].to_numpy())
    return iso.predict(_GRID)


def log_week(target: pd.DataFrame, season: int, week: int,
             n: float = DEFAULT_N, path: Path = LEDGER_PATH) -> list[dict]:
    """Log every qualifying ATS bet for the week. Idempotent per (season, week):
    if the week is already logged, returns its existing rows unchanged."""
    if target is None or target.empty:
        return []
    ledger = load_ledger(path)
    existing = ledger[(ledger["season"] == season) & (ledger["week"] == week)]
    if not existing.empty:
        return existing.to_dict("records")

    prob_grid = _prob_grid(season, week)
    rows = []
    for _, r in target.iterrows():
        if pd.isna(r.get("spread_line")) or pd.isna(r.get("model_home_prob")):
            continue
        p = min(max(float(r["model_home_prob"]), 1e-4), 1 - 1e-4)
        implied = float(np.interp(p, prob_grid, _GRID))
        gap = implied - float(r["spread_line"])
        if abs(gap) < n:
            continue
        bet_home = gap > 0
        rows.append({
            "season": int(season), "week": int(week), "game_id": str(r["game_id"]),
            "away_team": r["away_team"], "home_team": r["home_team"],
            "side": r["home_team"] if bet_home else r["away_team"],
            "bet_home": bool(bet_home),
            "model_home_prob": round(p, 4), "spread_line": float(r["spread_line"]),
            "implied_spread": round(implied, 2), "gap": round(gap, 2),
            "result": "open", "profit": np.nan,
            "logged_at": _now(), "settled_at": np.nan,
        })
    if rows:
        ledger = pd.concat([ledger, pd.DataFrame(rows)], ignore_index=True)
        _save(ledger, path)
    return rows


def settle(graded: pd.DataFrame, season: int,
           path: Path = LEDGER_PATH) -> pd.DataFrame:
    """Settle open ATS plays whose games are now final. ``graded`` is a
    grade.grade_season frame (game_id, home_score, away_score)."""
    ledger = load_ledger(path)
    if ledger.empty:
        return ledger
    margin = (graded.assign(m=graded["home_score"] - graded["away_score"])
              .set_index("game_id")["m"].to_dict())
    open_mask = (ledger["season"] == season) & (ledger["result"] == "open")
    for i in ledger[open_mask].index:
        gid = ledger.at[i, "game_id"]
        if gid not in margin or pd.isna(margin[gid]):
            continue
        m, line = float(margin[gid]), float(ledger.at[i, "spread_line"])
        if m == line:
            ledger.at[i, "result"], ledger.at[i, "profit"] = "push", 0.0
        else:
            bet_home = bool(ledger.at[i, "bet_home"])
            covered = (m > line) if bet_home else (m < line)
            ledger.at[i, "result"] = "win" if covered else "loss"
            ledger.at[i, "profit"] = STAKE * VIG_PAYOUT if covered else -STAKE
        ledger.at[i, "settled_at"] = _now()
    _save(ledger, path)
    return ledger


def summary(ledger: pd.DataFrame | None = None,
            path: Path = LEDGER_PATH) -> dict:
    """Running record + ROI over settled (win/loss) plays; pushes excluded."""
    if ledger is None:
        ledger = load_ledger(path)
    dec = ledger[ledger["result"].isin(["win", "loss"])]
    wins = int((dec["result"] == "win").sum())
    losses = int((dec["result"] == "loss").sum())
    staked = float(len(dec) * STAKE)
    profit = float(dec["profit"].sum()) if len(dec) else 0.0
    return {
        "bets": len(dec), "wins": wins, "losses": losses,
        "open": int((ledger["result"] == "open").sum()),
        "push": int((ledger["result"] == "push").sum()),
        "staked": staked, "profit": profit,
        "roi": profit / staked if staked else float("nan"),
    }


def render(season: int, week: int, path: Path = LEDGER_PATH) -> list[str]:
    """Markdown block for the grade report: running ATS ledger + this week."""
    ledger = load_ledger(path)
    if ledger.empty:
        return []
    s = summary(ledger)
    roi = f"{s['roi']:+.1%}" if s["bets"] else "—"
    lines = [
        "", "## 📐 Paper play — spread (ATS) tracker (out-of-sample)", "",
        f"**{s['wins']}-{s['losses']} settled · {s['profit']:+.1f}u on "
        f"{s['staked']:.0f}u staked · ROI {roi}** ({s['open']} open, "
        f"{s['push']} push)  ·  flat {STAKE:.0f}u -110, model's side ATS when its "
        f"implied spread diverges from the line by ≥{DEFAULT_N:g} pts. "
        f"Promising in backtest but unproven — tracking forward.",
        "",
    ]
    show = ledger[ledger["season"] == season]
    if show.empty:
        return lines
    lines += ["| Week | Side | Line | Gap | Result |", "|---|---|---|---|---|"]
    for _, r in show.sort_values(["week", "game_id"]).iterrows():
        line = f"{float(r['spread_line']):+g}"
        gap = f"{float(r['gap']):+.1f}" if pd.notna(r["gap"]) else "—"
        if r["result"] == "win":
            res = f"✓ +{float(r['profit']):.1f}u"
        elif r["result"] == "loss":
            res = f"✗ {float(r['profit']):.1f}u"
        elif r["result"] == "push":
            res = "push"
        else:
            res = "open"
        lines.append(f"| {int(r['week'])} | {r['side']} | {line} | {gap} | {res} |")
    return lines
