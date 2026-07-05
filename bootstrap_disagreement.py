"""Bootstrap the top-N disagreement strategy's ROI to see if it clears zero.

Reproduces the same walk-forward bets as backtest_disagreement.py (top-N
biggest model-vs-market disagreements/week, bet the model's favoured side,
optionally underdogs-only), collects the per-bet profit/stake outcomes, then
resamples them with replacement to get a confidence interval on ROI.

A strategy whose bootstrap ROI interval still straddles 0 is indistinguishable
from break-even at this sample size — i.e. the headline ROI could be variance.

Usage:
    ./.venv/bin/python bootstrap_disagreement.py --topn 1
    ./.venv/bin/python bootstrap_disagreement.py --topn 3 --dogs-only
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from grade import grade_season
from nfl_betting_model import data
from nfl_betting_model.betting import american_to_payout


def collect_bets(start: int, end: int, topn: int, stake: float,
                 kind: str, train_start: int, dogs_only: bool) -> pd.DataFrame:
    rows = []
    for season in range(start, end + 1):
        graded = grade_season(season, train_start=train_start, kind=kind)
        graded["edge"] = graded["model_home_prob"] - graded["market_home_prob"]
        games = data.load_games([season], include_unplayed=False)
        graded = graded.merge(
            games[["game_id", "home_moneyline", "away_moneyline"]],
            on="game_id", how="left", suffixes=("", "_g"))
        picks = (graded.assign(_abs=graded["edge"].abs())
                 .sort_values(["week", "_abs"], ascending=[True, False])
                 .groupby("week").head(topn))
        for _, r in picks.iterrows():
            bet_home = r["edge"] > 0
            ml = r["home_moneyline"] if bet_home else r["away_moneyline"]
            if pd.isna(ml):
                continue
            if dogs_only and ml <= 0:
                continue
            won = (r["home_win"] == 1) if bet_home else (r["home_win"] == 0)
            payout = american_to_payout(pd.Series([ml]))[0]
            profit = stake * payout if won else -stake
            rows.append({"season": season, "profit": profit, "staked": stake})
    return pd.DataFrame(rows)


def _summarise(label: str, rois: np.ndarray) -> None:
    lo, hi = np.percentile(rois, [2.5, 97.5])
    verdict = ("CLEARS zero — CI is entirely positive"
               if lo > 0 else
               "STRADDLES zero — indistinguishable from break-even")
    print(f"{label}:")
    print(f"  mean ROI          {rois.mean():+.1%}")
    print(f"  95% CI            [{lo:+.1%}, {hi:+.1%}]")
    print(f"  P(ROI > 0)        {float((rois > 0).mean()):.1%}")
    print(f"  5th pctile ROI    {np.percentile(rois, 5):+.1%}")
    print(f"  verdict:          {verdict}")


def bootstrap(bets: pd.DataFrame, n_boot: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    profit = bets["profit"].to_numpy()
    staked = bets["staked"].to_numpy()
    n = len(bets)
    obs_roi = profit.sum() / staked.sum()

    print(f"Bets: {n}  |  observed ROI: {obs_roi:+.1%}  |  "
          f"total profit {profit.sum():+.1f}u on {staked.sum():.0f}u staked\n")

    # 1) Bet-level bootstrap: resample individual bets (assumes iid).
    rois = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        rois[i] = profit[idx].sum() / staked[idx].sum()
    _summarise(f"Bet-level bootstrap ({n_boot:,} resamples)", rois)

    # 2) Season-block bootstrap: resample whole seasons with replacement,
    #    preserving within-season correlation (the honest test).
    groups = [(g["profit"].to_numpy(), g["staked"].to_numpy())
              for _, g in bets.groupby("season")]
    n_seasons = len(groups)
    rois_b = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n_seasons, n_seasons)
        p = np.concatenate([groups[j][0] for j in idx])
        s = np.concatenate([groups[j][1] for j in idx])
        rois_b[i] = p.sum() / s.sum()
    print()
    _summarise(f"Season-block bootstrap ({n_seasons} seasons, "
               f"{n_boot:,} resamples)", rois_b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2016)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--topn", type=int, default=1)
    ap.add_argument("--stake", type=float, default=10.0)
    ap.add_argument("--kind", default="logistic", choices=["logistic", "gbm"])
    ap.add_argument("--train-start", type=int, default=2010)
    ap.add_argument("--dogs-only", action="store_true")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"Collecting bets: top-{args.topn} disagreements, "
          f"{'dogs-only, ' if args.dogs_only else ''}"
          f"{args.start}-{args.end} ...\n")
    bets = collect_bets(args.start, args.end, args.topn, args.stake,
                        args.kind, args.train_start, args.dogs_only)
    bootstrap(bets, args.n_boot, args.seed)


if __name__ == "__main__":
    main()
