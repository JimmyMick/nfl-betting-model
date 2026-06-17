"""Experiment: bet only the single biggest model-vs-market disagreement each week.

Each week of a season we find the game where the model most disagrees with the
closing line (largest |model_home_prob - market_home_prob|), bet a flat 1-unit
moneyline on the side the model favours, and settle at the real American odds.
Then we tally the season.

This is a deliberately concentrated version of the betting question the project
already answered in aggregate (moneyline is efficient) — does the model's
*highest-conviction* disagreement each week fare any better?

    uv run experiment_topedge.py --season 2024
    uv run experiment_topedge.py --seasons 2019-2024 --kind logistic
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from grade import grade_season
from nfl_betting_model.betting import american_to_payout


def _favoured_side(edge: float) -> str:
    """'home' if the model likes the home side more than the market, else 'away'."""
    return "home" if edge > 0 else "away"


def run_season(season: int, train_start: int, kind: str, stake: float = 1.0) -> pd.DataFrame:
    """One row per week: the top-disagreement game, the bet, and its result."""
    s = grade_season(season, train_start=train_start, kind=kind)
    s = s.copy()
    s["edge"] = s["model_home_prob"] - s["market_home_prob"]
    s["fav"] = s["edge"].apply(_favoured_side)
    # Odds for the side we'd actually bet (the model's favoured side).
    s["bet_ml"] = np.where(s["fav"] == "home", s["home_moneyline"], s["away_moneyline"])
    # A bet is usable only if that side has a real price.
    s["bet_ml"] = pd.to_numeric(s["bet_ml"], errors="coerce")

    rows = []
    for week, g in s.groupby("week"):
        usable = g[g["bet_ml"].notna()]
        if usable.empty:
            continue
        # Biggest disagreement with a usable price.
        pick = usable.loc[usable["edge"].abs().idxmax()]
        won = (pick["home_win"] == 1) if pick["fav"] == "home" else (pick["home_win"] == 0)
        payout = american_to_payout(np.array([pick["bet_ml"]]))[0]
        profit = stake * payout if won else -stake
        fav_team = pick["home_team"] if pick["fav"] == "home" else pick["away_team"]
        rows.append({
            "week": int(week),
            "game": f"{pick['away_team']} @ {pick['home_team']}",
            "bet": fav_team,
            "ml": int(pick["bet_ml"]),
            "model": pick["model_home_prob"] if pick["fav"] == "home" else 1 - pick["model_home_prob"],
            "market": pick["market_home_prob"] if pick["fav"] == "home" else 1 - pick["market_home_prob"],
            "edge": abs(pick["edge"]),
            "won": bool(won),
            "profit": profit,
        })
    out = pd.DataFrame(rows)
    out["cum_profit"] = out["profit"].cumsum()
    out["season"] = season
    return out


def summarize(out: pd.DataFrame, stake: float = 1.0) -> dict:
    n = len(out)
    wins = int(out["won"].sum())
    staked = n * stake
    profit = out["profit"].sum()
    return {
        "bets": n,
        "record": f"{wins}-{n - wins}",
        "win_rate": wins / n if n else float("nan"),
        "staked": staked,
        "profit": profit,
        "roi": profit / staked if staked else float("nan"),
    }


def _fmt(out: pd.DataFrame) -> str:
    show = out.copy()
    show["model"] = (show["model"] * 100).round(0).astype(int).astype(str) + "%"
    show["market"] = (show["market"] * 100).round(0).astype(int).astype(str) + "%"
    show["edge"] = "+" + (show["edge"] * 100).round(0).astype(int).astype(str) + "%"
    show["ml"] = show["ml"].map(lambda v: f"{v:+d}")
    show["won"] = show["won"].map({True: "W", False: "L"})
    show["profit"] = show["profit"].round(2)
    show["cum_profit"] = show["cum_profit"].round(2)
    cols = ["week", "game", "bet", "ml", "model", "market", "edge", "won",
            "profit", "cum_profit"]
    return show[cols].to_string(index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int)
    ap.add_argument("--seasons", type=str, help="inclusive range, e.g. 2019-2024")
    ap.add_argument("--train-start", type=int, default=2010)
    ap.add_argument("--kind", choices=["logistic", "gbm"], default="logistic")
    ap.add_argument("--stake", type=float, default=1.0)
    args = ap.parse_args()

    if args.seasons:
        lo, hi = (int(x) for x in args.seasons.split("-"))
        seasons = list(range(lo, hi + 1))
    elif args.season:
        seasons = [args.season]
    else:
        raise SystemExit("pass --season or --seasons")

    all_rows = []
    for season in seasons:
        out = run_season(season, args.train_start, args.kind, args.stake)
        all_rows.append(out)
        summ = summarize(out, args.stake)
        print(f"\n=== {season} ({args.kind}) — bet the top disagreement each week ===")
        print(_fmt(out))
        print(f"\n  {summ['bets']} bets · {summ['record']} "
              f"({summ['win_rate']:.0%}) · staked {summ['staked']:.0f}u · "
              f"profit {summ['profit']:+.2f}u · ROI {summ['roi']:+.1%}")

    if len(seasons) > 1:
        pooled = pd.concat(all_rows, ignore_index=True)
        summ = summarize(pooled, args.stake)
        print(f"\n=== POOLED {seasons[0]}-{seasons[-1]} ({args.kind}) ===")
        print(f"  {summ['bets']} bets · {summ['record']} "
              f"({summ['win_rate']:.0%}) · staked {summ['staked']:.0f}u · "
              f"profit {summ['profit']:+.2f}u · ROI {summ['roi']:+.1%}")
        by_season = (pooled.groupby("season")["profit"].sum().round(2))
        print("\n  Per-season profit (units):")
        for season, p in by_season.items():
            print(f"    {season}: {p:+.2f}u")


if __name__ == "__main__":
    main()
