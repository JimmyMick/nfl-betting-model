"""Throwaway: multi-season walk-forward of the starter-talent features.

Re-measures the marginal value of the starter-unit Madden features
(ol/dl/db/starter_ovr_diff) AFTER the data-leak fix (starters are now defined
from prior-game snap share, not the game's own realized snaps). Tests ADD in two
bases:

  * ISOLATION (Elo+EPA)             — does it help beyond raw team strength?
  * FULL (Elo+EPA+QB+availability)  — does it help beyond the rest of the live
                                      model? (base + starters == the live config)

Sigmoid calibration, expanding-window train on every prior season — identical
discipline to validate_epa_splits.py. EPA + QB come from the parquet caches so
there is no play-by-play reload. Decision rule (project convention): a feature
earns its place only if it improves the majority of held-out seasons on
logloss/brier.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nfl_betting_model import (
    availability as avail_mod, data, model, starters as starters_mod)
from nfl_betting_model.elo import compute_elo
from nfl_betting_model.features import build_features

SEASONS = list(range(2010, 2026))
TEST_SEASONS = list(range(2021, 2026))
CALIBRATE = "sigmoid"
EPA_CACHE = Path("data/epa_cache")     # build_epa_cache.py
QB_CACHE = Path("data/full_cache")     # build_full_cache.py

print(f"Loading {SEASONS[0]}-{SEASONS[-1]} ...")
games = data.load_games(SEASONS)
elo = compute_elo(games)
epa = pd.concat(
    [pd.read_parquet(EPA_CACHE / f"{s}.parquet")[["game_id", "team", "off_epa",
                                                  "def_epa"]] for s in SEASONS],
    ignore_index=True)
qb = pd.concat([pd.read_parquet(QB_CACHE / f"qb_{s}.parquet") for s in SEASONS],
               ignore_index=True)
avail = avail_mod.team_out_talent(SEASONS)
start = starters_mod.starter_unit_ovr(SEASONS)   # leak-free (prior-game snaps)
print(f"  {len(games)} games | epa {len(epa)} | qb {len(qb)} | "
      f"avail {len(avail)} | starters {len(start)}")


def compare(name, base_kwargs, var_kwargs):
    df_b, cols_b = build_features(games, **base_kwargs)
    df_v, cols_v = build_features(games, **var_kwargs)
    print(f"\n=== {name} ({len(cols_b)}→{len(cols_v)} feats, {CALIBRATE}) ===")
    print(f"  {'season':>6}  {'logloss base→var':>22}  {'brier base→var':>20}  "
          f"{'auc base→var':>18}")
    wins = {"logloss": 0, "brier": 0, "auc": 0}
    for ts in TEST_SEASONS:
        tr_b, te_b = model.time_split(df_b, ts)
        tr_v, te_v = model.time_split(df_v, ts)
        r_b = model.evaluate(
            model.train(tr_b, cols_b, kind="logistic", calibrate=CALIBRATE), te_b, cols_b)
        r_v = model.evaluate(
            model.train(tr_v, cols_v, kind="logistic", calibrate=CALIBRATE), te_v, cols_v)
        ll, br, au = r_v.log_loss < r_b.log_loss, r_v.brier < r_b.brier, r_v.auc > r_b.auc
        wins["logloss"] += ll; wins["brier"] += br; wins["auc"] += au
        m = lambda b: "✓" if b else "✗"
        print(f"  {ts:>6}   {r_b.log_loss:.4f}→{r_v.log_loss:.4f} {m(ll)}   "
              f"{r_b.brier:.4f}→{r_v.brier:.4f} {m(br)}   "
              f"{r_b.auc:.4f}→{r_v.auc:.4f} {m(au)}")
    n = len(TEST_SEASONS)
    print(f"  improved:  logloss {wins['logloss']}/{n}  brier {wins['brier']}/{n}  "
          f"auc {wins['auc']}/{n}")


iso = dict(epa_table=epa, elo_table=elo)
compare("ADD / Isolation: Elo+EPA (+starters)", iso, dict(**iso, starter_table=start))

full = dict(epa_table=epa, elo_table=elo, qb_table=qb, avail_table=avail)
compare("ADD / Full: Elo+EPA+QB+avail (+starters == live)",
        full, dict(**full, starter_table=start))
