"""Cheap probe: turn the win-prob model into an implied spread and test it ATS.

No new model. For each season we take the trained win probabilities (via
grade_season), convert each to an implied home margin with a normal approx
(margin ~ Normal(mu, sigma), sigma = NFL margin SD ~13.5, so
mu = sigma * Phi^-1(P_home_win)), and then ask two things:

1. **Is the model a decent margin predictor?** Compare model-implied margin vs
   the market spread as predictors of the actual result (RMSE / MAE). The honest
   expectation is the closing spread wins.
2. **Any ATS signal?** Bet the side the model's implied margin favours relative
   to the spread, flat 1u at the real spread odds. Break-even at -110 is ~52.4%.

If margins track the line and ATS hovers at break-even, the spread market is
efficient for us too and a full margin model isn't worth building.

    uv run probe_spread.py --seasons 2016-2024
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import norm

import nflreadpy as nfl
from grade import grade_season
from nfl_betting_model.betting import american_to_payout

SIGMA = 13.5  # NFL final-margin standard deviation (well-established ~13-14).


def _schedule_extras(seasons: list[int]) -> pd.DataFrame:
    # spread_line + scores already come from data.load_games; we only need the
    # spread prices, which aren't in the model frame.
    sched = nfl.load_schedules().to_pandas()
    keep = ["game_id", "home_spread_odds", "away_spread_odds"]
    return sched[sched["season"].isin(seasons)][keep]


def run(seasons: list[int], train_start: int, kind: str) -> pd.DataFrame:
    extras = _schedule_extras(seasons)
    frames = []
    for season in seasons:
        s = grade_season(season, train_start=train_start, kind=kind)
        s = s.merge(extras, on="game_id", how="left")
        s = s[s["spread_line"].notna() & s["home_score"].notna()].copy()

        p = s["model_home_prob"].clip(0.01, 0.99)
        s["model_margin"] = SIGMA * norm.ppf(p)          # implied home margin
        # nflverse spread_line: positive = home favoured by that many points.
        s["mkt_margin"] = s["spread_line"]
        s["actual"] = s["home_score"] - s["away_score"]  # actual home margin

        # ATS: bet home to cover when the model expects home to beat the line.
        s["bet_home"] = s["model_margin"] > s["mkt_margin"]
        home_cover = s["actual"] > s["mkt_margin"]
        push = s["actual"] == s["mkt_margin"]
        s["won"] = np.where(s["bet_home"], home_cover, ~home_cover) & ~push
        s["push"] = push
        odds = np.where(s["bet_home"], s["home_spread_odds"], s["away_spread_odds"])
        odds = pd.to_numeric(pd.Series(odds), errors="coerce").fillna(-110).to_numpy()
        payout = american_to_payout(odds)
        s["profit"] = np.where(push, 0.0, np.where(s["won"], payout, -1.0))
        s["season"] = season
        frames.append(s)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", type=str, default="2016-2024")
    ap.add_argument("--train-start", type=int, default=2010)
    ap.add_argument("--kind", choices=["logistic", "gbm"], default="logistic")
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.seasons.split("-"))
    seasons = list(range(lo, hi + 1))

    df = run(seasons, args.train_start, args.kind)

    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def mae(a, b):
        return float(np.mean(np.abs(a - b)))

    print(f"\n=== Margin prediction, {lo}-{hi} ({args.kind}), n={len(df)} games ===")
    print(f"  model-implied margin : RMSE {rmse(df['model_margin'], df['actual']):.2f}  "
          f"MAE {mae(df['model_margin'], df['actual']):.2f}")
    print(f"  market spread        : RMSE {rmse(df['mkt_margin'], df['actual']):.2f}  "
          f"MAE {mae(df['mkt_margin'], df['actual']):.2f}")
    corr = np.corrcoef(df["model_margin"], df["mkt_margin"])[0, 1]
    print(f"  corr(model margin, market spread) = {corr:.3f}")
    print(f"  mean |model - market| = {mae(df['model_margin'], df['mkt_margin']):.2f} pts")

    bets = df[~df["push"]]
    n = len(bets)
    w = int(bets["won"].sum())
    roi = bets["profit"].sum() / n
    print(f"\n=== ATS (bet model's side vs the spread, flat 1u) ===")
    print(f"  {n} bets · {w}-{n - w} ({w / n:.1%}) · "
          f"break-even ~52.4% · profit {bets['profit'].sum():+.2f}u · ROI {roi:+.1%}")

    print("\n  Per-season ATS:")
    for season, g in bets.groupby("season"):
        gw = int(g["won"].sum())
        gn = len(g)
        print(f"    {season}: {gw}-{gn - gw} ({gw / gn:.0%}) · "
              f"ROI {g['profit'].sum() / gn:+.1%}")


if __name__ == "__main__":
    main()
