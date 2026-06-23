"""Throwaway: does extending the availability feature to QUESTIONABLE players
(at a calibrated weight) beat the current Out/Doubtful-only version?

Baseline avail = sum of weight*max(0,OVR-65) over Out(1.0)/Doubtful(0.75).
Variants add Questionable at a few weights. Same single feature
(`out_avail_diff`), different values — so this is a clean "is the upgraded
availability a better availability?" test, sigmoid walk-forward, expanding
window, identical discipline to validate_avail.py.

EPA comes from the per-season cache (build_epa_cache.py) so the ISOLATION run is
pbp-free and fast. FULL (+QB+Starters) is gated behind EPA_FULL=1 (QB loads pbp).

Run:  ./.venv/bin/python validate_avail_upgrade.py
"""

from __future__ import annotations

import os
from pathlib import Path

import nflreadpy as nfl
import pandas as pd

from nfl_betting_model import data, madden as madden_mod, model
from nfl_betting_model.elo import compute_elo
from nfl_betting_model.features import build_features

SEASONS = list(range(2010, 2026))
TEST_SEASONS = list(range(2021, 2026))
CALIBRATE = "sigmoid"
CACHE = Path("data/epa_cache")
REPL = 65
RUN_FULL = os.environ.get("EPA_FULL") == "1"

print(f"Loading {SEASONS[0]}-{SEASONS[-1]} ...")
games = data.load_games(SEASONS)
elo = compute_elo(games)

cached = pd.concat([pd.read_parquet(CACHE / f"{s}.parquet") for s in SEASONS],
                   ignore_index=True)
epa = cached[["game_id", "team", "off_epa", "def_epa"]].copy()

# Injuries (one season at a time; skip seasons without a feed) + Madden OVR.
frames = []
for s in SEASONS:
    try:
        one = nfl.load_injuries(seasons=[s])
    except ValueError:
        continue
    frames.append(one.to_pandas() if hasattr(one, "to_pandas") else pd.DataFrame(one))
inj_all = pd.concat(frames, ignore_index=True)[
    ["season", "week", "team", "gsis_id", "report_status"]]
inj_seasons = sorted(int(s) for s in inj_all["season"].dropna().unique())
ratings = madden_mod.load_ratings(inj_seasons)[["gsis_id", "season", "overallrating"]]

print("report_status counts:")
print(inj_all["report_status"].value_counts().to_string())

sched = pd.concat([
    games[["game_id", "season", "week", "home_team"]].rename(columns={"home_team": "team"}),
    games[["game_id", "season", "week", "away_team"]].rename(columns={"away_team": "team"}),
], ignore_index=True)


def avail_table(status_weights: dict[str, float]) -> pd.DataFrame:
    inj = inj_all[inj_all["report_status"].isin(status_weights)].copy()
    inj["weight"] = inj["report_status"].map(status_weights)
    inj = inj.merge(ratings, on=["gsis_id", "season"], how="left")
    above = (inj["overallrating"] - REPL).clip(lower=0).fillna(0.0)
    inj["contribution"] = inj["weight"] * above
    per = (inj.groupby(["season", "week", "team"])["contribution"].sum()
           .reset_index().rename(columns={"contribution": "out_avail"}))
    out = sched.merge(per, on=["season", "week", "team"], how="left")
    out["out_avail"] = out["out_avail"].fillna(0.0)
    return out[["game_id", "team", "out_avail"]]


BASE = {"Out": 1.0, "Doubtful": 0.75}
base_avail = avail_table(BASE)


def compare(name, base_kwargs, var_avail):
    df_b, cols_b = build_features(games, avail_table=base_avail, **base_kwargs)
    df_v, cols_v = build_features(games, avail_table=var_avail, **base_kwargs)
    print(f"\n=== {name} ({CALIBRATE}) ===")
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
SWEEP = tuple(float(x) for x in os.environ.get("QW", "0.15,0.20,0.30").split(","))
for w in SWEEP:
    var = avail_table({**BASE, "Questionable": w})
    compare(f"ISO Elo+EPA+avail: +Questionable@{w}", iso, var)

if RUN_FULL:
    from nfl_betting_model import starters as starters_mod
    FC = Path("data/full_cache")  # QB cache built by build_full_cache.py
    qb = pd.concat([pd.read_parquet(FC / f"qb_{s}.parquet") for s in SEASONS],
                   ignore_index=True)
    starters = starters_mod.starter_unit_ovr(SEASONS)  # snap-based, loaded live
    full = dict(epa_table=epa, elo_table=elo, qb_table=qb, starter_table=starters)
    for w in SWEEP:
        var = avail_table({**BASE, "Questionable": w})
        compare(f"FULL Elo+EPA+QB+Starters+avail: +Questionable@{w}", full, var)
