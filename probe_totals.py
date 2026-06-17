"""Cheap probe: is there ANY edge predicting game totals vs the closing O/U line?

Deliberately simple and independent of the win-prob model (which says nothing
about totals). For each game we build leak-free rolling team scoring features
(each team's points-for and points-against over its prior games, shifted so the
current game is excluded), fit a small linear regression on all *earlier*
seasons (expanding), and predict the game's total points. Then:

1. **Forecast quality** — model predicted-total RMSE/MAE vs the market total
   line as predictors of the actual total. Honest expectation: the line wins.
2. **O/U signal** — bet Over when the model predicts more than the line, Under
   when less, flat 1u at the real over/under odds. Break-even ~52.4% at -110.

If the line forecasts better and O/U hovers at break-even, totals is efficient
for a simple model and a full totals build isn't worth it.

    uv run probe_totals.py --seasons 2016-2024
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import nflreadpy as nfl
from nfl_betting_model.betting import american_to_payout

WINDOW = 10        # rolling window of prior team games
MIN_GAMES = 4      # need at least this much history to use a game
FEATURES = ["home_pf", "home_pa", "away_pf", "away_pa", "dome"]


def _build() -> pd.DataFrame:
    s = nfl.load_schedules().to_pandas()
    s = s[s["home_score"].notna()].copy()
    s["gameday"] = pd.to_datetime(s["gameday"])
    s = s.sort_values("gameday").reset_index(drop=True)
    s["actual_total"] = s["home_score"] + s["away_score"]
    s["dome"] = s["roof"].isin(["dome", "closed"]).astype(int)

    # Long team-game log to compute each team's rolling points for/against.
    home = s[["game_id", "gameday", "home_team", "home_score", "away_score"]].rename(
        columns={"home_team": "team", "home_score": "pf", "away_score": "pa"})
    away = s[["game_id", "gameday", "away_team", "away_score", "home_score"]].rename(
        columns={"away_team": "team", "away_score": "pf", "home_score": "pa"})
    log = pd.concat([home, away], ignore_index=True).sort_values(["team", "gameday"])

    g = log.groupby("team")
    # shift(1) so a game never sees its own result (leak-free).
    log["roll_pf"] = g["pf"].transform(
        lambda x: x.shift(1).rolling(WINDOW, min_periods=MIN_GAMES).mean())
    log["roll_pa"] = g["pa"].transform(
        lambda x: x.shift(1).rolling(WINDOW, min_periods=MIN_GAMES).mean())

    roll = log[["game_id", "team", "roll_pf", "roll_pa"]]
    h = roll.rename(columns={"team": "home_team", "roll_pf": "home_pf",
                             "roll_pa": "home_pa"})
    a = roll.rename(columns={"team": "away_team", "roll_pf": "away_pf",
                             "roll_pa": "away_pa"})
    s = s.merge(h, on=["game_id", "home_team"]).merge(a, on=["game_id", "away_team"])
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", type=str, default="2016-2024")
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.seasons.split("-"))
    seasons = list(range(lo, hi + 1))

    s = _build()
    s = s.dropna(subset=FEATURES + ["actual_total"])

    preds = []
    for season in seasons:
        train = s[s["season"] < season]
        test = s[s["season"] == season].copy()
        if train.empty or test.empty:
            continue
        reg = LinearRegression().fit(train[FEATURES], train["actual_total"])
        test["pred_total"] = reg.predict(test[FEATURES])
        preds.append(test)
    df = pd.concat(preds, ignore_index=True)
    g = df.dropna(subset=["total_line"]).copy()

    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def mae(a, b):
        return float(np.mean(np.abs(a - b)))

    print(f"\n=== Total prediction, {lo}-{hi}, n={len(g)} games ===")
    print(f"  model predicted total : RMSE {rmse(g['pred_total'], g['actual_total']):.2f}  "
          f"MAE {mae(g['pred_total'], g['actual_total']):.2f}")
    print(f"  market total line     : RMSE {rmse(g['total_line'], g['actual_total']):.2f}  "
          f"MAE {mae(g['total_line'], g['actual_total']):.2f}")
    corr = np.corrcoef(g["pred_total"], g["total_line"])[0, 1]
    print(f"  corr(model total, market line) = {corr:.3f}")
    print(f"  mean |model - market| = {mae(g['pred_total'], g['total_line']):.2f} pts")

    # O/U bet: Over when model > line, Under when model < line.
    g["bet_over"] = g["pred_total"] > g["total_line"]
    over_hit = g["actual_total"] > g["total_line"]
    push = g["actual_total"] == g["total_line"]
    g["won"] = np.where(g["bet_over"], over_hit, ~over_hit) & ~push
    odds = np.where(g["bet_over"], g["over_odds"], g["under_odds"])
    odds = pd.to_numeric(pd.Series(odds), errors="coerce").fillna(-110).to_numpy()
    payout = american_to_payout(odds)
    g["profit"] = np.where(push, 0.0, np.where(g["won"], payout, -1.0))

    bets = g[~push]
    n = len(bets)
    w = int(bets["won"].sum())
    print(f"\n=== Over/Under (bet model's side vs the line, flat 1u) ===")
    print(f"  {n} bets · {w}-{n - w} ({w / n:.1%}) · break-even ~52.4% · "
          f"profit {bets['profit'].sum():+.2f}u · ROI {bets['profit'].sum() / n:+.1%}")
    print("\n  Per-season O/U:")
    for season, gs in bets.groupby("season"):
        gw = int(gs["won"].sum())
        gn = len(gs)
        print(f"    {season}: {gw}-{gn - gw} ({gw / gn:.0%}) · "
              f"ROI {gs['profit'].sum() / gn:+.1%}")


if __name__ == "__main__":
    main()
