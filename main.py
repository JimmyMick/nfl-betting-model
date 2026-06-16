"""Train and evaluate the NFL moneyline win-probability model.

Runs an ablation across feature sets (base form -> +Elo -> +Elo+EPA) and model
types, all benchmarked against the Vegas moneyline on the same holdout season.

Examples
--------
    uv run main.py --train 2010-2022 --test 2023
    uv run main.py --train 2016-2022 --test 2023 --no-epa
"""

from __future__ import annotations

import argparse

from nfl_betting_model import (
    betting, coaching as coaching_mod, data, epa as epa_mod, model,
    qb as qb_mod, qb_epa as qb_epa_mod, starters as starters_mod,
)
from nfl_betting_model.elo import compute_elo
from nfl_betting_model.features import build_features


def _parse_range(text: str) -> range:
    if "-" in text:
        lo, hi = (int(x) for x in text.split("-", 1))
        return range(lo, hi + 1)
    year = int(text)
    return range(year, year + 1)


def _run(games, epa_table, elo_table, test_season, label, kind, calibrate=None,
         bets=False, qb_table=None, starter_table=None, coach_table=None,
         qb_epa_table=None):
    df, cols = build_features(
        games, epa_table=epa_table, elo_table=elo_table, qb_table=qb_table,
        qb_epa_table=qb_epa_table, starter_table=starter_table,
        coach_table=coach_table,
    )
    train_df, test_df = model.time_split(df, test_season)
    pipe = model.train(train_df, cols, kind=kind, calibrate=calibrate)
    result = model.evaluate(pipe, test_df, cols)
    cal_tag = f", {calibrate}-calibrated" if calibrate else ""
    print(f"\n[{label}]  ({len(cols)} features, {kind}{cal_tag})")
    print(result)
    if bets:
        print(betting.betting_report(test_df, result.home_prob))
    return result


EV_THRESHOLDS = (0.0, 0.02, 0.05, 0.10)


def _backtest(games, epa_table, elo_table, test_seasons, calibrate, history_start,
              qb_table=None):
    """Walk-forward backtest: each test season is predicted by a model trained
    on every season before it (expanding window). Bets are pooled across seasons.
    """
    tag = f"{calibrate}-calibrated" if calibrate else "uncalibrated"
    qb_tag = " +QB" if qb_table is not None else ""
    print(f"\n=== Walk-forward backtest, gbm{qb_tag} ({tag}) ===")
    print(f"  train: expanding from {history_start};  test seasons: "
          f"{test_seasons[0]}-{test_seasons[-1]}")

    df, cols = build_features(
        games, epa_table=epa_table, elo_table=elo_table, qb_table=qb_table
    )
    per_threshold: dict[float, list] = {t: [] for t in EV_THRESHOLDS}

    print(f"\n  {'season':>6}  {'acc':>5}  {'mkt':>5}   bets(EV>=2%)  ROI(EV>=2%)")
    for ts in test_seasons:
        train_df, test_df = model.time_split(df, ts)
        if train_df.empty or test_df.empty:
            continue
        pipe = model.train(train_df, cols, kind="gbm", calibrate=calibrate)
        result = model.evaluate(pipe, test_df, cols)
        for t in EV_THRESHOLDS:
            per_threshold[t].append(
                betting.evaluate_betting(test_df, result.home_prob, ev_threshold=t)
            )
        r2 = betting.evaluate_betting(test_df, result.home_prob, ev_threshold=0.02)
        print(f"  {ts:>6}  {result.accuracy:.3f}  {result.market_accuracy:.3f}   "
              f"{r2.n_bets:>10d}   {r2.roi:>+8.1%}")

    print("\n  Pooled across all test seasons (flat 1u, +EV side):")
    for t in EV_THRESHOLDS:
        print(betting.combine(per_threshold[t]))


def main() -> None:
    ap = argparse.ArgumentParser(description="NFL moneyline win-probability model")
    ap.add_argument("--train", default="2010-2022")
    ap.add_argument("--test", default="2023")
    ap.add_argument("--no-epa", action="store_true", help="skip play-by-play EPA")
    ap.add_argument("--no-madden", action="store_true",
                    help="skip starting-QB Madden OVR feature")
    ap.add_argument("--no-starters", action="store_true",
                    help="skip starting-unit (OL/DL/secondary) OVR features")
    ap.add_argument("--no-coaching", action="store_true",
                    help="skip coaching (career win%% + new-regime) features")
    ap.add_argument("--no-qb-epa", action="store_true",
                    help="skip per-player rolling QB EPA/play feature")
    ap.add_argument(
        "--calibrate",
        choices=["isotonic", "sigmoid"],
        default=None,
        help="calibrate the GBM probabilities (time-aware holdout)",
    )
    ap.add_argument(
        "--backtest",
        default=None,
        metavar="START-END",
        help="walk-forward backtest over these test seasons (e.g. 2019-2023); "
             "trains expanding window from --train start",
    )
    args = ap.parse_args()

    if args.backtest:
        test_seasons = list(_parse_range(args.backtest))
        history_start = list(_parse_range(args.train))[0]
        seasons = list(range(history_start, test_seasons[-1] + 1))
    else:
        seasons = list(_parse_range(args.train)) + [int(args.test)]
    test_season = int(args.test)

    print(f"Loading schedules {seasons[0]}-{seasons[-1]} ...")
    games = data.load_games(seasons)
    elo_table = compute_elo(games)
    print(f"  {len(games)} games, Elo computed")

    epa_table = None
    if not args.no_epa:
        print("Loading play-by-play EPA (one season at a time) ...")
        epa_table = epa_mod.team_game_epa(seasons)
        print(f"  {len(epa_table)} team-game EPA rows")

    qb_table = None
    if not args.no_madden:
        print("Loading starting-QB Madden OVR ...")
        qb_table = qb_mod.starting_qb_ovr(seasons)
        rated = qb_table["qb_ovr"].notna().sum()
        print(f"  {len(qb_table)} game-team QBs, {rated} with a rating")

    starter_table = None
    if not args.no_starters:
        print("Loading starting-unit (OL/DL/secondary) Madden OVR ...")
        starter_table = starters_mod.starter_unit_ovr(seasons)
        print(f"  {len(starter_table)} game-team starting-unit rows")

    coach_table = None
    if not args.no_coaching:
        print("Loading coaching features (career win% + new-regime) ...")
        coach_table = coaching_mod.coach_features(seasons)
        rated = coach_table["coach_winpct"].notna().sum()
        print(f"  {len(coach_table)} game-team coach rows, {rated} with a prior record")

    qb_epa_table = None
    if not args.no_qb_epa:
        print("Loading per-player rolling QB EPA/play ...")
        qb_epa_table = qb_epa_mod.starting_qb_epa(seasons)
        rated = qb_epa_table["qb_epa"].notna().sum()
        print(f"  {len(qb_epa_table)} game-team starters, {rated} with prior EPA")

    if args.backtest:
        # Best feature set both with and without the QB rating, to isolate it.
        _backtest(games, epa_table, elo_table, test_seasons, "isotonic", seasons[0])
        if qb_table is not None:
            _backtest(games, epa_table, elo_table, test_seasons, "isotonic",
                      seasons[0], qb_table=qb_table)
        return

    print("\n=== Ablation (test season "
          f"{test_season}, benchmarked vs market) ===")
    _run(games, None, None, test_season, "base form", "logistic")
    _run(games, None, elo_table, test_season, "base + Elo", "logistic")
    if epa_table is not None:
        _run(games, epa_table, elo_table, test_season, "base + Elo + EPA", "logistic")
        # Best model: report uncalibrated, calibrated, and ROI vs the market.
        _run(games, epa_table, elo_table, test_season, "base + Elo + EPA", "gbm",
             bets=True)
        _run(games, epa_table, elo_table, test_season, "base + Elo + EPA", "gbm",
             calibrate="isotonic", bets=True)
        _run(games, epa_table, elo_table, test_season, "base + Elo + EPA", "gbm",
             calibrate="sigmoid", bets=True)

        if qb_table is not None:
            # Layer the starting-QB Madden OVR on top of the best feature set.
            _run(games, epa_table, elo_table, test_season,
                 "base + Elo + EPA + QB", "logistic", qb_table=qb_table)
            _run(games, epa_table, elo_table, test_season,
                 "base + Elo + EPA + QB", "gbm", bets=True, qb_table=qb_table)
            _run(games, epa_table, elo_table, test_season,
                 "base + Elo + EPA + QB", "gbm", calibrate="isotonic",
                 bets=True, qb_table=qb_table)

        if qb_table is not None and starter_table is not None:
            # Full player-level set: QB + starting-unit (OL/DL/secondary) OVR.
            # Pivot focus = probability quality (logloss/brier), not ROI.
            _run(games, epa_table, elo_table, test_season,
                 "+ QB + Starters (OL/DL/DB)", "logistic",
                 qb_table=qb_table, starter_table=starter_table)
            _run(games, epa_table, elo_table, test_season,
                 "+ QB + Starters (OL/DL/DB)", "gbm",
                 qb_table=qb_table, starter_table=starter_table)

        if qb_epa_table is not None:
            # Per-player rolling QB EPA isolated on team strength (Elo+EPA):
            # does QB-attributed EPA add over team off_epa it overlaps with?
            _run(games, epa_table, elo_table, test_season,
                 "base + Elo + EPA + QBepa", "logistic", qb_epa_table=qb_epa_table)
            _run(games, epa_table, elo_table, test_season,
                 "base + Elo + EPA + QBepa", "gbm", qb_epa_table=qb_epa_table)

            if qb_table is not None:
                # Head-to-head/with the static Madden QB OVR: complement or
                # redundant? (rolling performance vs fixed preseason talent).
                _run(games, epa_table, elo_table, test_season,
                     "base + Elo + EPA + QB(Madden) + QBepa", "logistic",
                     qb_table=qb_table, qb_epa_table=qb_epa_table)
                _run(games, epa_table, elo_table, test_season,
                     "base + Elo + EPA + QB(Madden) + QBepa", "gbm",
                     qb_table=qb_table, qb_epa_table=qb_epa_table)

        if coach_table is not None:
            # Coaching isolated on top of team strength (Elo+EPA): the cleanest
            # read on whether it adds anything orthogonal to team quality.
            _run(games, epa_table, elo_table, test_season,
                 "base + Elo + EPA + Coaching", "logistic", coach_table=coach_table)
            _run(games, epa_table, elo_table, test_season,
                 "base + Elo + EPA + Coaching", "gbm", coach_table=coach_table)

            if qb_table is not None and starter_table is not None:
                # Coaching layered on the full player-level set.
                _run(games, epa_table, elo_table, test_season,
                     "+ QB + Starters + Coaching", "logistic", qb_table=qb_table,
                     starter_table=starter_table, coach_table=coach_table)
                _run(games, epa_table, elo_table, test_season,
                     "+ QB + Starters + Coaching", "gbm", qb_table=qb_table,
                     starter_table=starter_table, coach_table=coach_table)


if __name__ == "__main__":
    main()
