"""Throwaway: compare calibration strategies on the full feature set.

The production path calibrates isotonically on the *single* latest training
season (FrozenEstimator), which is noisy and tanks when that season is anomalous
(2020 COVID -> 2021). Compare, per test season 2021-2025 (train = all prior):

  uncal      - no calibration (logistic is often already decent)
  sigmoid_1  - current mechanism, Platt on the last season
  isotonic_1 - current PRODUCTION: isotonic on the last season
  sigmoid_cv - Platt, 5-fold CV over ALL training data
  isotonic_cv- isotonic, 5-fold CV over ALL training data

Reports per-season log loss + the mean and std (stability) across seasons.
"""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from nfl_betting_model import (
    data, epa as epa_mod, model, qb as qb_mod, starters as starters_mod,
)
from nfl_betting_model.elo import compute_elo
from nfl_betting_model.features import build_features, market_home_prob
from nfl_betting_model.model import build_pipeline

SEASONS = list(range(2010, 2026))
TEST_SEASONS = list(range(2021, 2026))

print(f"Loading {SEASONS[0]}-{SEASONS[-1]} ...")
games = data.load_games(SEASONS)
elo = compute_elo(games)
epa = epa_mod.team_game_epa(SEASONS)
qb = qb_mod.starting_qb_ovr(SEASONS)
starters = starters_mod.starter_unit_ovr(SEASONS)
df, cols = build_features(games, epa_table=epa, elo_table=elo,
                          qb_table=qb, starter_table=starters)
print(f"  {len(df)} modelled games, {len(cols)} features")


def fit_uncal(tr):
    p = build_pipeline("logistic"); p.fit(tr[cols], tr["home_win"]); return p

def fit_last(tr, method):
    return model.train(tr, cols, kind="logistic", calibrate=method)

def fit_cv(tr, method):
    c = CalibratedClassifierCV(build_pipeline("logistic"), method=method, cv=5)
    c.fit(tr[cols], tr["home_win"]); return c


VARIANTS = {
    "uncal":       lambda tr: fit_uncal(tr),
    "sigmoid_1":   lambda tr: fit_last(tr, "sigmoid"),
    "isotonic_1":  lambda tr: fit_last(tr, "isotonic"),
    "sigmoid_cv":  lambda tr: fit_cv(tr, "sigmoid"),
    "isotonic_cv": lambda tr: fit_cv(tr, "isotonic"),
}

results = {name: {"ll": [], "br": [], "auc": []} for name in VARIANTS}
mkt_ll = []
for ts in TEST_SEASONS:
    tr, te = model.time_split(df, ts)
    y = te["home_win"].to_numpy()
    mkt = market_home_prob(te).to_numpy()
    mkt_ll.append(log_loss(y, mkt, labels=[0, 1]))
    for name, fit in VARIANTS.items():
        p = fit(tr).predict_proba(te[cols])[:, 1]
        results[name]["ll"].append(log_loss(y, p, labels=[0, 1]))
        results[name]["br"].append(brier_score_loss(y, p))
        results[name]["auc"].append(roc_auc_score(y, p))

print(f"\n  market mean log loss: {np.mean(mkt_ll):.4f}")
print(f"\n  {'variant':>12}  {'logloss/season (2021..2025)':>38}  "
      f"{'mean':>6} {'std':>6}  {'brier':>6}  {'auc':>6}")
for name in VARIANTS:
    ll = results[name]["ll"]
    per = " ".join(f"{x:.3f}" for x in ll)
    print(f"  {name:>12}  {per:>38}  {np.mean(ll):.4f} {np.std(ll):.4f}  "
          f"{np.mean(results[name]['br']):.4f}  {np.mean(results[name]['auc']):.4f}")
