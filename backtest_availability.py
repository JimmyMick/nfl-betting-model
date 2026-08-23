"""Strategy C1: availability-driven bets (reviews/additional_betting_strategies.md).

Availability is the one feature that independently clears the validation bar and is
orthogonal to team strength (who's literally not suiting up). This tests whether
betting the model's side ONLY when availability is what drives the disagreement is
purer signal than betting all disagreements.

Decomposition: train the FULL model (Elo+EPA+QB+avail) and a BASE model
(Elo+EPA+QB, no avail) on the same walk-forward. For each game,
``avail_shift = full_prob - base_prob`` is availability's push on the model, and
``edge = full_prob - market_prob`` is the disagreement with the market. Bet the
full model's favoured side, flat 10u, only when availability is a MATERIAL and
ALIGNED contributor to the disagreement:

    sign(avail_shift) == sign(edge)   AND   |avail_shift| >= TAU

TAU is pre-registered at 0.03 (a ~3-point availability move, roughly a key starter
ruled out) — NOT tuned on results (guardrail #3). Judged by season-block bootstrap.
Caches only (no pbp reload).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import bootstrap_disagreement as boot
from nfl_betting_model import availability as avail_mod, data, model
from nfl_betting_model.betting import american_to_payout
from nfl_betting_model.elo import compute_elo
from nfl_betting_model.features import build_features, market_home_prob

SEASONS = list(range(2010, 2026))
TEST_SEASONS = list(range(2019, 2026))
STAKE, TAU, CAL = 10.0, 0.03, "sigmoid"   # TAU pre-registered, not tuned
EPA_CACHE, QB_CACHE = Path("data/epa_cache"), Path("data/full_cache")

print(f"Loading {SEASONS[0]}-{SEASONS[-1]} ...")
games = data.load_games(SEASONS)
elo = compute_elo(games)
epa = pd.concat([pd.read_parquet(EPA_CACHE / f"{s}.parquet")[
    ["game_id", "team", "off_epa", "def_epa"]] for s in SEASONS], ignore_index=True)
qb = pd.concat([pd.read_parquet(QB_CACHE / f"qb_{s}.parquet") for s in SEASONS],
               ignore_index=True)
avail = avail_mod.team_out_talent(SEASONS)

full_kw = dict(epa_table=epa, elo_table=elo, qb_table=qb, avail_table=avail)
base_kw = dict(epa_table=epa, elo_table=elo, qb_table=qb)           # no availability
df_f, cols_f = build_features(games, **full_kw)   # already carries moneyline cols
df_b, cols_b = build_features(games, **base_kw)

rows = []
for ts in TEST_SEASONS:
    full = model.train(df_f[df_f["season"] < ts], cols_f, kind="logistic", calibrate=CAL)
    base = model.train(df_b[df_b["season"] < ts], cols_b, kind="logistic", calibrate=CAL)
    te_f = df_f[df_f["season"] == ts].copy()
    te_b = df_b[df_b["season"] == ts]
    te_f["full_p"] = full.predict_proba(te_f[cols_f])[:, 1]
    te_f["base_p"] = base.predict_proba(te_b[cols_b])[:, 1]
    te_f["mkt_p"] = market_home_prob(te_f).to_numpy()
    te_f = te_f.dropna(subset=["mkt_p", "home_win"])
    te_f["avail_shift"] = te_f["full_p"] - te_f["base_p"]
    te_f["edge"] = te_f["full_p"] - te_f["mkt_p"]
    # availability materially + directionally drives the disagreement
    pick = te_f[(np.sign(te_f["avail_shift"]) == np.sign(te_f["edge"]))
                & (te_f["avail_shift"].abs() >= TAU)]
    for _, r in pick.iterrows():
        bet_home = r["full_p"] >= 0.5
        m = r["home_moneyline"] if bet_home else r["away_moneyline"]
        if pd.isna(m):
            continue
        won = (r["home_win"] == 1) if bet_home else (r["home_win"] == 0)
        profit = STAKE * american_to_payout(pd.Series([m]))[0] if won else -STAKE
        rows.append({"season": ts, "week": int(r["week"]),
                     "bet": r["home_team"] if bet_home else r["away_team"],
                     "ml": int(m), "avail_shift": round(float(r["avail_shift"]), 3),
                     "won": bool(won), "profit": round(float(profit), 2),
                     "staked": STAKE})

d = pd.DataFrame(rows)
print(f"\nStrategy C1 — availability-driven bets (|avail_shift|>= {TAU}, aligned), "
      f"2019-2025\n")
print(f"{'Season':>6} {'Bets':>5} {'W':>3} {'L':>3} {'Win%':>6} {'Profit':>8} {'ROI':>7}")
for s, g in d.groupby("season"):
    b = len(g); w = int(g["won"].sum()); p = g["profit"].sum()
    print(f"{s:>6} {b:>5} {w:>3} {b-w:>3} {w/b:>6.0%} {p:>+8.1f} {p/(b*STAKE):>+7.1%}")
print(f"\nTotal bets {len(d)} | avg ML {d['ml'].mean():+.0f} | "
      f"dogs(+ML) {int((d['ml']>0).sum())}/{len(d)}\n")
boot.bootstrap(d[["season", "profit", "staked"]], n_boot=10000, seed=0)
