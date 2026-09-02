"""Resumable per-week top-1 disagreement ledger across a season range.

Runs one season per invocation (or as many as fit), appending rows to
predictions/wide_ledger.csv so repeated runs pick up where they left off.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from grade import grade_season
from nfl_betting_model.betting import american_to_payout

STAKE, TOPN, TRAIN_START = 10.0, 1, 2010
OUT = Path("predictions/wide_ledger.csv")
SEASONS = list(range(2019, 2026))


def season_ledger(season: int) -> pd.DataFrame:
    graded = grade_season(season, train_start=TRAIN_START, kind="logistic")
    graded["edge"] = graded["model_home_prob"] - graded["market_home_prob"]
    picks = (graded.assign(_abs=graded["edge"].abs())
             .sort_values(["week", "_abs"], ascending=[True, False])
             .groupby("week").head(TOPN))
    rows = []
    for _, r in picks.iterrows():
        bet_home = r["edge"] > 0
        side = r["home_team"] if bet_home else r["away_team"]
        ml = r["home_moneyline"] if bet_home else r["away_moneyline"]
        if pd.isna(ml):
            continue
        won = (r["home_win"] == 1) if bet_home else (r["home_win"] == 0)
        payout = american_to_payout(pd.Series([ml]))[0]  # net fractional odds b
        profit = STAKE * payout if won else -STAKE
        winner = r["home_team"] if r["home_win"] == 1 else r["away_team"]
        # Bet-side model + market-implied win probabilities (for Kelly etc.).
        model_p = r["model_home_prob"] if bet_home else 1 - r["model_home_prob"]
        market_p = r["market_home_prob"] if bet_home else 1 - r["market_home_prob"]
        rows.append({
            "season": season, "week": int(r["week"]),
            "matchup": f"{r['away_team']} @ {r['home_team']}",
            "bet": side, "ml": int(ml), "b": round(float(payout), 4),
            "edge": abs(r["edge"]), "model_p": round(float(model_p), 4),
            "market_p": round(float(market_p), 4),
            "winner": winner, "won": bool(won), "profit": round(profit, 2),
        })
    return pd.DataFrame(rows)


def main() -> None:
    done = set()
    if OUT.exists():
        prev = pd.read_csv(OUT)
        done = set(prev["season"].unique().tolist())
    todo = [s for s in SEASONS if s not in done]
    if not todo:
        print("All seasons already computed.")
        return
    # One season per run keeps each invocation under the time limit.
    season = todo[0]
    print(f"Computing {season} ({len(todo)} remaining: {todo}) ...")
    led = season_ledger(season)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    led.to_csv(OUT, mode="a", header=not OUT.exists() or OUT.stat().st_size == 0,
               index=False)
    w = int(led["won"].sum())
    n = len(led)
    print(f"{season}: {n} bets, {w}-{n - w}, profit {led['profit'].sum():+.1f}u")
    remaining = [s for s in todo if s != season]
    print(f"REMAINING={remaining}")
    sys.exit(0 if not remaining else 7)  # 7 = more to do


if __name__ == "__main__":
    main()
