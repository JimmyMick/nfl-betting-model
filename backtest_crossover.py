"""Resumable backtest: bet the model's side when model and market pick DIFFERENT
winners (a "pick crossover" — the model likes the team the market has as the
underdog to win outright).

For each game: model favours home iff model_home_prob >= .5; market favours home
iff market_home_prob >= .5. When those disagree, stake 10u flat on the model's
favoured side at that side's closing moneyline. Uses the current (leaner) model.

One season per invocation, appending to predictions/crossover_ledger.csv, so a
kill mid-run resumes. Re-run until it prints "All seasons already computed."
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from grade import grade_season
from nfl_betting_model.betting import american_to_payout

STAKE, TRAIN_START = 10.0, 2010
OUT = Path("predictions/crossover_ledger.csv")
SEASONS = list(range(2019, 2026))


def season_rows(season: int) -> pd.DataFrame:
    g = grade_season(season, train_start=TRAIN_START, kind="logistic")
    model_home = g["model_home_prob"] >= 0.5
    market_home = g["market_home_prob"] >= 0.5
    cross = g[model_home != market_home].copy()      # pick disagreement
    rows = []
    for _, r in cross.iterrows():
        bet_home = r["model_home_prob"] >= 0.5       # model's favoured side
        ml = r["home_moneyline"] if bet_home else r["away_moneyline"]
        if pd.isna(ml):
            continue
        won = (r["home_win"] == 1) if bet_home else (r["home_win"] == 0)
        payout = american_to_payout(pd.Series([ml]))[0]
        profit = STAKE * payout if won else -STAKE
        side = r["home_team"] if bet_home else r["away_team"]
        winner = r["home_team"] if r["home_win"] == 1 else r["away_team"]
        rows.append({
            "season": season, "week": int(r["week"]),
            "matchup": f"{r['away_team']} @ {r['home_team']}",
            "bet": side, "ml": int(ml),
            "model_p": round(float(max(r["model_home_prob"], 1 - r["model_home_prob"])), 3),
            "market_p": round(float(max(r["market_home_prob"], 1 - r["market_home_prob"])), 3),
            "winner": winner, "won": bool(won), "profit": round(float(profit), 2),
        })
    return pd.DataFrame(rows)


def main() -> None:
    done = set()
    if OUT.exists():
        done = set(pd.read_csv(OUT)["season"].unique().tolist())
    todo = [s for s in SEASONS if s not in done]
    if not todo:
        print("All seasons already computed.")
        return
    season = todo[0]
    print(f"Computing {season} ({len(todo)} remaining: {todo}) ...")
    rows = season_rows(season)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(OUT, mode="a", header=not OUT.exists() or OUT.stat().st_size == 0,
                index=False)
    w = int(rows["won"].sum()); n = len(rows)
    print(f"{season}: {n} crossover bets, {w}-{n - w}, profit {rows['profit'].sum():+.1f}u")
    remaining = [s for s in todo if s != season]
    print(f"REMAINING={remaining}")
    sys.exit(0 if not remaining else 7)


if __name__ == "__main__":
    main()
