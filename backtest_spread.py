"""Strategy E1: spread (ATS) betting from model-vs-market disagreement.

The moneyline market is efficient for this model (§7d/§7f), but spreads are a
structurally different market with lower, symmetric vig (-110 ~ 4.5% hold vs
5-9% favourite ML juice). This tests whether a marginal edge survives ATS.

Mapping (key-number-aware, refined): convert the model's calibrated win prob to
an implied home spread by inverting the EMPIRICAL spread->P(home win) curve. An
isotonic regression is fit on TRAINING seasons only (leak-free), preserving the
win-prob jumps around key numbers (3, 7) that a normal approximation smooths
over; it is then inverted on a half-point spread grid.
gap = implied - spread_line. Bet HOME ATS when gap >= +N (model favours home by
more than the line), AWAY ATS when gap <= -N. Settle at -110 vs the actual
margin (home covers iff home_margin > spread_line; ties push).

Resumable: one season per invocation -> predictions/spread_ledger.csv, storing
every qualifying bet at the minimum threshold (|gap| >= 1) so any N can be sliced
later. Run repeatedly until "All seasons already computed."
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from grade import grade_season
from nfl_betting_model import data

STAKE, TRAIN_START, VIG_PAYOUT = 10.0, 2010, 100.0 / 110.0   # -110 both sides
MIN_GAP = 1.0                                                  # store from here
OUT = Path("predictions/spread_emp_ledger.csv")
SEASONS = list(range(2019, 2026))
_GRID = np.arange(-21.0, 21.0001, 0.5)                         # half-point spreads

_ALL = data.load_games(list(range(TRAIN_START, 2026)))
_ALL = _ALL.assign(home_margin=_ALL["home_score"] - _ALL["away_score"])


def _implied_spread(model_probs: pd.Series, season: int) -> np.ndarray:
    """Key-number-aware prob -> implied home spread, fit on seasons < season."""
    tr = _ALL[(_ALL["season"] < season) & _ALL["home_win"].notna()]
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(tr["spread_line"].to_numpy(), tr["home_win"].to_numpy())
    prob_grid = iso.predict(_GRID)                     # P(home win) at each spread
    # Invert: interpolate the spread whose empirical win prob equals model prob.
    p = model_probs.clip(1e-4, 1 - 1e-4).to_numpy()
    return np.interp(p, prob_grid, _GRID)


def season_rows(season: int) -> pd.DataFrame:
    g = grade_season(season, train_start=TRAIN_START, kind="logistic")
    g = g.assign(home_margin=g["home_score"] - g["away_score"]).dropna(
        subset=["spread_line", "home_margin", "model_home_prob"])

    implied = _implied_spread(g["model_home_prob"], season)
    g["gap"] = implied - g["spread_line"]
    g = g[g["gap"].abs() >= MIN_GAP]

    rows = []
    for _, r in g.iterrows():
        bet_home = r["gap"] > 0
        margin, line = r["home_margin"], r["spread_line"]
        if margin == line:
            result, profit, staked = "push", 0.0, 0.0
        else:
            covered = (margin > line) if bet_home else (margin < line)
            result = "win" if covered else "loss"
            profit = STAKE * VIG_PAYOUT if covered else -STAKE
            staked = STAKE
        rows.append({
            "season": season, "week": int(r["week"]),
            "side": r["home_team"] if bet_home else r["away_team"],
            "abs_gap": round(float(abs(r["gap"])), 2),
            "spread_line": float(line), "home_margin": float(margin),
            "result": result, "profit": round(float(profit), 3),
            "staked": staked})
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
    dec = rows[rows["result"] != "push"]
    w = int((dec["result"] == "win").sum())
    print(f"{season}: {len(dec)} bets ({len(rows)-len(dec)} push), "
          f"{w}-{len(dec)-w}, profit {rows['profit'].sum():+.1f}u")
    remaining = [s for s in todo if s != season]
    print(f"REMAINING={remaining}")
    sys.exit(0 if not remaining else 7)


if __name__ == "__main__":
    main()
