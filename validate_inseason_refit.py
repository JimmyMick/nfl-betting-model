"""Throwaway: does adding in-season games to TRAINING help, or just add noise?

Compares two training disciplines on the live feature set (Elo+EPA+QB+avail),
walk-forward over held-out seasons, strictly leak-free:

  * BASELINE (current)     — fit once on every season BEFORE the target, then
                             predict every week of the target season.
  * WEEKLY REFIT (variant) — before each week w, refit on all prior seasons PLUS
                             that season's weeks 1…w-1, then predict week w.

By default both arms use the shipped sigmoid calibration; the refit arm passes
``calib_season = ts-1`` so the in-season rows train the BASE model (not just the
Platt calibrator). Pass ``none`` as the 2nd arg to run uncalibrated (isolates the
base-model effect). Same games scored in both arms. EPA/QB from caches (no pbp).

Usage: ./.venv/bin/python validate_inseason_refit.py [logistic|gbm] [sigmoid|none]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from nfl_betting_model import availability as avail_mod, data, model
from nfl_betting_model.elo import compute_elo
from nfl_betting_model.features import build_features

KIND = sys.argv[1] if len(sys.argv) > 1 else "logistic"
CAL = None if (len(sys.argv) > 2 and sys.argv[2] == "none") else "sigmoid"
SEASONS = list(range(2010, 2026))
TEST_SEASONS = list(range(2021, 2026))
EPA_CACHE, QB_CACHE = Path("data/epa_cache"), Path("data/full_cache")

print(f"[{KIND}] Loading {SEASONS[0]}-{SEASONS[-1]} ...")
games = data.load_games(SEASONS)
elo = compute_elo(games)
epa = pd.concat([pd.read_parquet(EPA_CACHE / f"{s}.parquet")[
    ["game_id", "team", "off_epa", "def_epa"]] for s in SEASONS], ignore_index=True)
qb = pd.concat([pd.read_parquet(QB_CACHE / f"qb_{s}.parquet") for s in SEASONS],
               ignore_index=True)
avail = avail_mod.team_out_talent(SEASONS)
df, cols = build_features(games, epa_table=epa, elo_table=elo, qb_table=qb,
                          avail_table=avail)
df = df[df["home_win"].notna()].copy()   # played games only


def _fit_predict(train_df, test_df, calib_season=None):
    pipe = model.train(train_df, cols, kind=KIND, calibrate=CAL,
                       calib_season=calib_season)
    return pipe.predict_proba(test_df[cols])[:, 1]


def _metrics(y, p):
    return (log_loss(y, p, labels=[0, 1]), brier_score_loss(y, p),
            roc_auc_score(y, p))


print(f"\n{'season':>6}  {'logloss base→refit':>24}  {'brier base→refit':>22}  "
      f"{'auc base→refit':>20}")
wins = {"ll": 0, "br": 0, "au": 0}
for ts in TEST_SEASONS:
    prior = df[df["season"] < ts]
    ts_df = df[df["season"] == ts].sort_values("week")
    y = ts_df["home_win"].to_numpy()

    # BASELINE: one model fit on prior seasons, applied to the whole season.
    # (calib defaults to the latest prior season, ts-1 — the shipped behaviour.)
    p_base = _fit_predict(prior, ts_df)

    # WEEKLY REFIT: expand the training set by each completed week. Hold out the
    # most recent complete season (ts-1) for calibration so the in-season rows
    # train the base model.
    p_refit = np.empty(len(ts_df))
    for w in sorted(ts_df["week"].unique()):
        tr = df[(df["season"] < ts) | ((df["season"] == ts) & (df["week"] < w))]
        wk_mask = (ts_df["week"] == w).to_numpy()
        p_refit[wk_mask] = _fit_predict(tr, ts_df[ts_df["week"] == w],
                                        calib_season=ts - 1)

    ll_b, br_b, au_b = _metrics(y, p_base)
    ll_r, br_r, au_r = _metrics(y, p_refit)
    ll, br, au = ll_r < ll_b, br_r < br_b, au_r > au_b
    wins["ll"] += ll; wins["br"] += br; wins["au"] += au
    m = lambda b: "✓" if b else "✗"
    print(f"  {ts:>6}   {ll_b:.4f}→{ll_r:.4f} {m(ll)}   "
          f"{br_b:.4f}→{br_r:.4f} {m(br)}   {au_b:.4f}→{au_r:.4f} {m(au)}")

n = len(TEST_SEASONS)
print(f"\n[{KIND}] refit improved:  logloss {wins['ll']}/{n}  "
      f"brier {wins['br']}/{n}  auc {wins['au']}/{n}")
