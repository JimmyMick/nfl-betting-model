"""Resumable backtest: FADE the model's favorite over-confidence.

Consensus games (model and market favour the same team) where the model is MORE
confident in that favourite than the market is (model_p > market_p). The prior
test showed *betting* that favourite loses ~-7% (the model over-rates chalk the
market prices sharper). This flips it: bet 10u on the UNDERDOG (the side opposite
the consensus favourite) at the dog's moneyline. Uses the current model.

One season per invocation, appending to predictions/dogfade_ledger.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from grade import grade_season
from nfl_betting_model.betting import american_to_payout

STAKE, TRAIN_START = 10.0, 2010
OUT = Path("predictions/dogfade_ledger.csv")
SEASONS = list(range(2019, 2026))


def season_rows(season: int) -> pd.DataFrame:
    g = grade_season(season, train_start=TRAIN_START, kind="logistic")
    model_home = g["model_home_prob"] >= 0.5
    market_home = g["market_home_prob"] >= 0.5
    g = g[model_home == market_home].copy()                 # consensus
    g["model_p"] = g[["model_home_prob"]].assign(
        o=1 - g["model_home_prob"]).max(axis=1)
    g["market_p"] = g[["market_home_prob"]].assign(
        o=1 - g["market_home_prob"]).max(axis=1)
    g = g[g["model_p"] > g["market_p"]]                     # model over-rates fav
    rows = []
    for _, r in g.iterrows():
        fav_home = r["model_home_prob"] >= 0.5
        dog_is_home = not fav_home                          # bet the OTHER side
        ml = r["home_moneyline"] if dog_is_home else r["away_moneyline"]
        if pd.isna(ml):
            continue
        won = (r["home_win"] == 1) if dog_is_home else (r["home_win"] == 0)
        payout = american_to_payout(pd.Series([ml]))[0]
        profit = STAKE * payout if won else -STAKE
        dog = r["home_team"] if dog_is_home else r["away_team"]
        winner = r["home_team"] if r["home_win"] == 1 else r["away_team"]
        rows.append({
            "season": season, "week": int(r["week"]),
            "matchup": f"{r['away_team']} @ {r['home_team']}",
            "bet_dog": dog, "ml": int(ml),
            "model_p": round(float(r["model_p"]), 3),
            "market_p": round(float(r["market_p"]), 3),
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
    print(f"{season}: {n} dog-fade bets, {w}-{n - w}, profit {rows['profit'].sum():+.1f}u")
    remaining = [s for s in todo if s != season]
    print(f"REMAINING={remaining}")
    sys.exit(0 if not remaining else 7)


if __name__ == "__main__":
    main()
