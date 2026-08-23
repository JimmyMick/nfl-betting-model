"""Throwaway A/B: leaky vs leak-free starter features.

Runs the same walk-forward as validate_starters.py but for BOTH definitions of a
"starter": the old leaky one (>=50% snaps in the SAME game) and the fixed one
(>=50% averaged over PRIOR games only). Quantifies how much held-out lift the
leak was fabricating. EPA/QB from caches (no pbp reload).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from nfl_betting_model import (
    availability as avail_mod, data, madden as madden_mod, model,
    starters as starters_mod)
from nfl_betting_model.elo import compute_elo
from nfl_betting_model.features import build_features

SEASONS = list(range(2010, 2026))
TEST_SEASONS = list(range(2021, 2026))
CAL = "sigmoid"
EPA_CACHE, QB_CACHE = Path("data/epa_cache"), Path("data/full_cache")


def leaky_starter_unit_ovr(seasons):
    """The pre-fix definition: starters chosen from the SAME game's snap pct."""
    sc = starters_mod._snap_counts(seasons)
    ratings = madden_mod.ratings_by_pfr(seasons)[["pfr_id", "season", "overallrating"]]
    sc = sc.merge(ratings, left_on=["pfr_player_id", "season"],
                  right_on=["pfr_id", "season"], how="left")
    off_start = sc["offense_pct"] >= starters_mod._SNAP_THRESHOLD  # same-game (leak)
    def_start = sc["defense_pct"] >= starters_mod._SNAP_THRESHOLD

    def _unit(mask, positions):
        sub = sc[mask]
        if positions is not None:
            sub = sub[sub["position"].isin(positions)]
        return sub.groupby(["game_id", "team"])["overallrating"].mean()

    return pd.DataFrame({
        "ol_ovr": _unit(off_start, starters_mod.OL_POS),
        "dl_ovr": _unit(def_start, starters_mod.DL_POS),
        "db_ovr": _unit(def_start, starters_mod.DB_POS),
        "starter_ovr": _unit(off_start | def_start, None),
    }).reset_index()


print(f"Loading {SEASONS[0]}-{SEASONS[-1]} ...")
games = data.load_games(SEASONS)
elo = compute_elo(games)
epa = pd.concat([pd.read_parquet(EPA_CACHE / f"{s}.parquet")[
    ["game_id", "team", "off_epa", "def_epa"]] for s in SEASONS], ignore_index=True)
qb = pd.concat([pd.read_parquet(QB_CACHE / f"qb_{s}.parquet") for s in SEASONS],
               ignore_index=True)
avail = avail_mod.team_out_talent(SEASONS)
leaky = leaky_starter_unit_ovr(SEASONS)
fixed = starters_mod.starter_unit_ovr(SEASONS)
print(f"  leaky rows {len(leaky)} | fixed rows {len(fixed)}")

full = dict(epa_table=epa, elo_table=elo, qb_table=qb, avail_table=avail)
df_b, cols_b = build_features(games, **full)
df_l, cols_l = build_features(games, **dict(**full, starter_table=leaky))
df_f, cols_f = build_features(games, **dict(**full, starter_table=fixed))

print("\nFull base (Elo+EPA+QB+avail). base→+starters, per held-out season.")
print(f"  {'season':>6} {'base LL':>8} {'leaky LL':>9} {'fixed LL':>9} "
      f"{'leaky Δ':>9} {'fixed Δ':>9}")
win_l = win_f = 0
for ts in TEST_SEASONS:
    (trb, teb) = model.time_split(df_b, ts)
    (trl, tel) = model.time_split(df_l, ts)
    (trf, tef) = model.time_split(df_f, ts)
    rb = model.evaluate(model.train(trb, cols_b, kind="logistic", calibrate=CAL), teb, cols_b)
    rl = model.evaluate(model.train(trl, cols_l, kind="logistic", calibrate=CAL), tel, cols_l)
    rf = model.evaluate(model.train(trf, cols_f, kind="logistic", calibrate=CAL), tef, cols_f)
    dl, df_ = rl.log_loss - rb.log_loss, rf.log_loss - rb.log_loss
    win_l += rl.log_loss < rb.log_loss
    win_f += rf.log_loss < rb.log_loss
    print(f"  {ts:>6} {rb.log_loss:>8.4f} {rl.log_loss:>9.4f} {rf.log_loss:>9.4f} "
          f"{dl:>+9.4f} {df_:>+9.4f}")
n = len(TEST_SEASONS)
print(f"\n  logloss-improved vs base:  LEAKY {win_l}/{n}   FIXED {win_f}/{n}")
print("  (Δ = starter-model logloss minus base; negative = better. The gap "
      "between 'leaky Δ' and 'fixed Δ' is the lift the leak was fabricating.)")
