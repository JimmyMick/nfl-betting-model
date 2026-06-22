"""Throwaway: multi-season walk-forward of opponent-adjusted early-down EPA.

Tests the richer EPA flavour from ``epa_oa.py`` two ways, each in an ISOLATION
(Elo+EPA) and a FULL (+QB+Starters, the live config) base:

  * REPLACE — swap raw all-down EPA for opponent-adjusted early-down EPA.
              "Is it a *better* EPA?"
  * ADD     — keep raw EPA and add opponent-adjusted early-down EPA beside it.
              "Does it carry value *orthogonal* to raw EPA?"

Sigmoid calibration, expanding-window train on every prior season — identical
discipline to validate_avail.py. Decision rule (project convention): a feature
ships only if it improves the majority of held-out seasons on logloss/brier.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from nfl_betting_model import data, epa_oa, model
from nfl_betting_model.elo import compute_elo
from nfl_betting_model.features import build_features

SEASONS = list(range(2010, 2026))
TEST_SEASONS = list(range(2021, 2026))
CALIBRATE = "sigmoid"
CACHE = Path("data/epa_cache")  # built by build_epa_cache.py (single pbp pass)
# FULL tests need the QB+Starters tables; QB loads pbp (slow). Default to the
# pbp-free ISOLATION run; set EPA_FULL=1 to also run the FULL comparisons.
RUN_FULL = os.environ.get("EPA_FULL") == "1"

print(f"Loading {SEASONS[0]}-{SEASONS[-1]} ...")
games = data.load_games(SEASONS)
elo = compute_elo(games)

# EPA from the per-season cache (raw all-down + raw early-down) — no pbp reload.
cached = pd.concat([pd.read_parquet(CACHE / f"{s}.parquet") for s in SEASONS],
                   ignore_index=True)
epa = cached[["game_id", "team", "off_epa", "def_epa"]].copy()
raw_ed = cached[["game_id", "team", "off_raw_ed", "def_raw_ed"]].rename(
    columns={"off_raw_ed": "off_raw", "def_raw_ed": "def_raw"})
oa = epa_oa.adjust_early_down(raw_ed, games)
print(f"  {len(games)} games | raw-EPA rows {len(epa)} | OA-ED rows {len(oa)}")


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

# REPLACE: raw EPA -> opponent-adjusted early-down EPA.
compare("REPLACE / Isolation: Elo+EPA",
        iso, dict(epa_table=oa, elo_table=elo))
# ADD: keep raw EPA, add opponent-adjusted early-down EPA beside it.
compare("ADD / Isolation: Elo+EPA (+OA-ED)",
        iso, dict(**iso, epa2_table=oa))

if RUN_FULL:
    from nfl_betting_model import qb as qb_mod, starters as starters_mod
    qb = qb_mod.starting_qb_ovr(SEASONS)
    starters = starters_mod.starter_unit_ovr(SEASONS)
    full = dict(epa_table=epa, elo_table=elo, qb_table=qb, starter_table=starters)
    compare("REPLACE / Full: +QB+Starters",
            full, dict(epa_table=oa, elo_table=elo, qb_table=qb, starter_table=starters))
    compare("ADD / Full: +QB+Starters (+OA-ED)",
            full, dict(**full, epa2_table=oa))
