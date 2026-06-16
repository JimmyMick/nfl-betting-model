"""Throwaway: multi-season walk-forward of the availability feature.

For each test season, compares (with vs without the injury-report availability
feature) on calibration metrics, on two base feature sets:
  - ISOLATION: Elo+EPA only (is the signal real?)
  - FULL: + QB + Starters (does it survive next to the snap-count starter
    features, which already encode some availability? — this is what the live
    predict.py uses)
Trains on every prior season (expanding window).
"""

from __future__ import annotations

from nfl_betting_model import (
    availability as avail_mod, data, epa as epa_mod, model,
    qb as qb_mod, starters as starters_mod,
)
from nfl_betting_model.elo import compute_elo
from nfl_betting_model.features import build_features

SEASONS = list(range(2010, 2026))
TEST_SEASONS = list(range(2021, 2026))

print(f"Loading {SEASONS[0]}-{SEASONS[-1]} ...")
games = data.load_games(SEASONS)
elo = compute_elo(games)
epa = epa_mod.team_game_epa(SEASONS)
avail = avail_mod.team_out_talent(SEASONS)
qb = qb_mod.starting_qb_ovr(SEASONS)
starters = starters_mod.starter_unit_ovr(SEASONS)
print(f"  {len(games)} games")


CALIBRATE = "sigmoid"  # match the production _train_for setup exactly


def compare(name, base_kwargs):
    df_b, cols_b = build_features(games, **base_kwargs)
    df_a, cols_a = build_features(games, avail_table=avail, **base_kwargs)
    print(f"\n=== {name} ({len(cols_b)}→{len(cols_a)} feats, {CALIBRATE}) ===")
    print(f"  {'season':>6}  {'logloss base→+avail':>22}  {'brier base→+avail':>20}  "
          f"{'auc base→+avail':>18}")
    wins = {"logloss": 0, "brier": 0, "auc": 0}
    for ts in TEST_SEASONS:
        tr_b, te_b = model.time_split(df_b, ts)
        tr_a, te_a = model.time_split(df_a, ts)
        r_b = model.evaluate(
            model.train(tr_b, cols_b, kind="logistic", calibrate=CALIBRATE), te_b, cols_b)
        r_a = model.evaluate(
            model.train(tr_a, cols_a, kind="logistic", calibrate=CALIBRATE), te_a, cols_a)
        ll, br, au = r_a.log_loss < r_b.log_loss, r_a.brier < r_b.brier, r_a.auc > r_b.auc
        wins["logloss"] += ll; wins["brier"] += br; wins["auc"] += au
        m = lambda b: "✓" if b else "✗"
        print(f"  {ts:>6}   {r_b.log_loss:.4f}→{r_a.log_loss:.4f} {m(ll)}   "
              f"{r_b.brier:.4f}→{r_a.brier:.4f} {m(br)}   "
              f"{r_b.auc:.4f}→{r_a.auc:.4f} {m(au)}")
    n = len(TEST_SEASONS)
    print(f"  improved:  logloss {wins['logloss']}/{n}  brier {wins['brier']}/{n}  "
          f"auc {wins['auc']}/{n}")


compare("ISOLATION: Elo+EPA", dict(epa_table=epa, elo_table=elo))
compare("FULL: + QB + Starters", dict(epa_table=epa, elo_table=elo,
                                      qb_table=qb, starter_table=starters))
