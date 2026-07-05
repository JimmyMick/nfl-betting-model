"""Backtest: flat-stake the top-N biggest model-vs-market disagreements each week.

Strategy under test (Jim's question):
  For every week of the last N seasons, rank that week's games by the model's
  disagreement with the market — |edge|, where edge = model_home_prob minus
  market_home_prob. Take the top 3 disagreements and bet a flat stake (default
  10 units) on the **model's favoured side** (the team the model likes more than
  the market does). Settle at the closing moneyline.

  This is exactly the family of bet the project already retired (moneyline ROI
  is robustly negative — the market is efficient), so treat the ROI here as a
  falsification check, not a green light.

Runs one training pass per season via grade.grade_season (train on every prior
season, grade the target), joins the closing moneylines from data.load_games,
then settles. Needs the nflverse schedules feed (github.com) for results +
odds — the EPA/QB/Madden feature caches on disk are not enough on their own.

Usage:
    ./.venv/bin/python backtest_disagreement.py                 # 2016-2025, top 3, 10u
    ./.venv/bin/python backtest_disagreement.py --start 2019 --end 2025 --topn 3 --stake 10
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from grade import grade_season
from nfl_betting_model import data
from nfl_betting_model.betting import american_to_payout


def _settle(picks: pd.DataFrame, stake: float, side: str,
            dogs_only: bool = False) -> dict:
    """Settle a set of bets. ``side`` = 'model' (favoured side) or 'market'.

    Returns per-bucket totals. Games without a usable moneyline for the chosen
    side are dropped (can't price the bet). If ``dogs_only``, keep only bets
    where the backed side is a market underdog (positive American moneyline).
    """
    rows = []
    for _, r in picks.iterrows():
        if side == "model":
            bet_home = r["edge"] > 0            # model's favoured side
        else:
            bet_home = r["market_home_prob"] >= 0.5  # market favourite
        ml = r["home_moneyline"] if bet_home else r["away_moneyline"]
        if pd.isna(ml):
            continue
        if dogs_only and ml <= 0:               # backed side isn't an underdog
            continue
        won = (r["home_win"] == 1) if bet_home else (r["home_win"] == 0)
        payout = american_to_payout(pd.Series([ml]))[0]  # profit multiplier per unit
        profit = stake * payout if won else -stake
        rows.append({"won": bool(won), "profit": profit, "staked": stake})
    if not rows:
        return {"bets": 0, "wins": 0, "staked": 0.0, "profit": 0.0}
    d = pd.DataFrame(rows)
    return {"bets": len(d), "wins": int(d["won"].sum()),
            "staked": float(d["staked"].sum()), "profit": float(d["profit"].sum())}


def _fmt(name: str, agg: dict) -> str:
    if not agg["bets"]:
        return f"{name}: no priced bets"
    roi = agg["profit"] / agg["staked"] if agg["staked"] else float("nan")
    wr = agg["wins"] / agg["bets"]
    return (f"{name}: {agg['bets']} bets, {agg['wins']}-{agg['bets'] - agg['wins']} "
            f"({wr:.1%})  staked {agg['staked']:.0f}u  "
            f"profit {agg['profit']:+.1f}u  ROI {roi:+.1%}")


def backtest(start: int, end: int, topn: int, stake: float,
             kind: str, train_start: int, dogs_only: bool = False) -> None:
    seasons = list(range(start, end + 1))
    dog_note = " (UNDERDOGS ONLY)" if dogs_only else ""
    print(f"Backtest: top-{topn} model-vs-market disagreements/week, "
          f"{stake:.0f}u flat, {kind} model, {seasons[0]}-{seasons[-1]}{dog_note}\n")

    per_season = []
    tot_model = {"bets": 0, "wins": 0, "staked": 0.0, "profit": 0.0}
    tot_market = {"bets": 0, "wins": 0, "staked": 0.0, "profit": 0.0}

    for season in seasons:
        graded = grade_season(season, train_start=train_start, kind=kind)
        graded["edge"] = graded["model_home_prob"] - graded["market_home_prob"]
        # Attach closing moneylines by game_id.
        games = data.load_games([season], include_unplayed=False)
        graded = graded.merge(
            games[["game_id", "home_moneyline", "away_moneyline"]],
            on="game_id", how="left", suffixes=("", "_g"))
        for c in ("home_moneyline", "away_moneyline"):
            if c not in graded and f"{c}_g" in graded:
                graded[c] = graded[f"{c}_g"]

        # Each week: take the top-N |edge| games.
        picks = (graded.assign(_abs=graded["edge"].abs())
                 .sort_values(["week", "_abs"], ascending=[True, False])
                 .groupby("week").head(topn))

        m = _settle(picks, stake, "model", dogs_only=dogs_only)
        k = _settle(picks, stake, "market", dogs_only=dogs_only)
        for tot, one in ((tot_model, m), (tot_market, k)):
            for key in tot:
                tot[key] += one[key]
        roi = m["profit"] / m["staked"] if m["staked"] else float("nan")
        per_season.append((season, m["bets"], m["wins"], m["profit"], roi))
        print(f"  {season}: {_fmt('model-side', m)}")

    print("\nPer-season (model's favoured side):")
    print(f"  {'season':>6} {'bets':>5} {'wins':>5} {'profit(u)':>10} {'ROI':>8}")
    for s, b, w, p, roi in per_season:
        print(f"  {s:>6} {b:>5} {w:>5} {p:>+10.1f} {roi:>+8.1%}")

    print("\n" + "=" * 60)
    print("TOTALS")
    print("  " + _fmt("Bet the MODEL's favoured side ", tot_model))
    print("  " + _fmt("Bet the MARKET favourite      ", tot_market))
    print("=" * 60)
    print("\nNote: moneyline betting was retired from this project — negative ROI\n"
          "is the expected, market-efficient result. Treat a negative number as\n"
          "confirmation, not a surprise.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest top-N disagreement bets")
    ap.add_argument("--start", type=int, default=2016)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--topn", type=int, default=3)
    ap.add_argument("--stake", type=float, default=10.0)
    ap.add_argument("--kind", default="logistic", choices=["logistic", "gbm"])
    ap.add_argument("--train-start", type=int, default=2010)
    ap.add_argument("--dogs-only", action="store_true",
                    help="only bet when the backed side is a market underdog")
    args = ap.parse_args()
    backtest(args.start, args.end, args.topn, args.stake, args.kind,
             args.train_start, dogs_only=args.dogs_only)


if __name__ == "__main__":
    main()
