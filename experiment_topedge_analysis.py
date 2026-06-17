"""Is the 'bet the biggest disagreement each week' edge real, or just variance?

For each season we train once (via grade_season) and, from the same scored
games, compare three flat-1u moneyline strategies on the model's favoured side:

* **top-edge**   — the single biggest |model - market| disagreement each week.
* **random**     — a random usable game each week (analytic expectation, i.e.
                   the average game; this is the fair baseline for "does the
                   *selection* matter?").
* **full slate** — bet model's side on every usable game (the known aggregate
                   null, shown for reference).

Then a bootstrap CI on the pooled top-edge per-bet profit tells us whether the
ROI is distinguishable from zero. Also splits top-edge profit by favourite vs
underdog to expose any favourite-longshot dependence.

    uv run experiment_topedge_analysis.py --seasons 2016-2024
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from grade import grade_season
from nfl_betting_model.betting import american_to_payout


def _score(season: int, train_start: int, kind: str) -> pd.DataFrame:
    """One row per usable game with the model-side bet's profit (flat 1u)."""
    s = grade_season(season, train_start=train_start, kind=kind).copy()
    s["edge"] = s["model_home_prob"] - s["market_home_prob"]
    s["home_fav"] = s["edge"] > 0
    s["bet_ml"] = pd.to_numeric(
        np.where(s["home_fav"], s["home_moneyline"], s["away_moneyline"]),
        errors="coerce")
    s = s[s["bet_ml"].notna()].copy()
    won = np.where(s["home_fav"], s["home_win"] == 1, s["home_win"] == 0)
    payout = american_to_payout(s["bet_ml"].to_numpy())
    s["profit"] = np.where(won, payout, -1.0)
    s["abs_edge"] = s["edge"].abs()
    s["is_dog"] = s["bet_ml"] > 0
    s["season"] = season
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", type=str, default="2016-2024")
    ap.add_argument("--train-start", type=int, default=2010)
    ap.add_argument("--kind", choices=["logistic", "gbm"], default="logistic")
    ap.add_argument("--boot", type=int, default=20000)
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.seasons.split("-"))
    seasons = list(range(lo, hi + 1))

    rng = np.random.default_rng(0)
    top_profits = []          # pooled per-bet profit of the top-edge strategy
    top_frames = []           # full top-pick rows (for dog/fav split)
    rows = []
    for season in seasons:
        s = _score(season, args.train_start, args.kind)
        weeks = s.groupby("week")
        top = weeks.apply(lambda g: g.loc[g["abs_edge"].idxmax()], include_groups=False)
        rnd_exp = weeks["profit"].mean().sum()         # E[random one game/week]
        n_weeks = weeks.ngroups
        full_roi = s["profit"].sum() / len(s)
        top_profits.append(top["profit"].to_numpy())
        top_frames.append(top[["profit", "is_dog", "bet_ml"]])
        rows.append({
            "season": season, "weeks": n_weeks,
            "top_profit": top["profit"].sum(), "top_roi": top["profit"].mean(),
            "rnd_profit": rnd_exp, "rnd_roi": rnd_exp / n_weeks,
            "full_roi": full_roi,
        })

    df = pd.DataFrame(rows)
    print(f"\nPer-season ROI ({args.kind}), flat 1u on the model's favoured side:\n")
    show = df.copy()
    for c in ["top_roi", "rnd_roi", "full_roi"]:
        show[c] = (show[c] * 100).round(1).astype(str) + "%"
    show["top_profit"] = show["top_profit"].round(2)
    show["rnd_profit"] = show["rnd_profit"].round(2)
    print(show[["season", "weeks", "top_profit", "top_roi",
                "rnd_roi", "full_roi"]].to_string(index=False))

    pooled = np.concatenate(top_profits)
    n = len(pooled)
    roi = pooled.mean()
    # Bootstrap CI on per-bet profit (ROI).
    boot = rng.choice(pooled, size=(args.boot, n), replace=True).mean(axis=1)
    lo_ci, hi_ci = np.percentile(boot, [2.5, 97.5])
    p_le0 = (boot <= 0).mean()

    rnd_pooled_roi = df["rnd_profit"].sum() / df["weeks"].sum()
    full_pooled_roi = df["full_roi"].mean()

    print(f"\n=== POOLED {lo}-{hi} ({args.kind}) ===")
    print(f"  top-edge : {n} bets · profit {pooled.sum():+.2f}u · "
          f"ROI {roi:+.1%} · 95% CI [{lo_ci:+.1%}, {hi_ci:+.1%}] · "
          f"P(ROI<=0)={p_le0:.1%}")
    print(f"  random   : ROI {rnd_pooled_roi:+.1%}  (fair baseline — does selection help?)")
    print(f"  full slate: ROI {full_pooled_roi:+.1%}  (aggregate null, reference)")

    # Favourite vs underdog dependence of the top-edge bets.
    allt = pd.concat(top_frames, ignore_index=True)
    dogs, favs = allt[allt["is_dog"]], allt[~allt["is_dog"]]
    print(f"\n  Top-edge bet mix: {len(dogs)} underdogs "
          f"(profit {dogs['profit'].sum():+.2f}u, ROI {dogs['profit'].mean():+.1%}) · "
          f"{len(favs)} favourites "
          f"(profit {favs['profit'].sum():+.2f}u, ROI {favs['profit'].mean():+.1%})")
    print("\n  Read: if the random baseline is also positive, the 'edge' is the "
          "week's games / odds in general, not the disagreement selection. If the "
          "CI includes 0, the result isn't distinguishable from variance.")


if __name__ == "__main__":
    main()
