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

from nfl_betting_model import data, epa as epa_mod, model
from nfl_betting_model.elo import compute_elo
from nfl_betting_model.features import build_features


def _parse_range(text: str) -> range:
    if "-" in text:
        lo, hi = (int(x) for x in text.split("-", 1))
        return range(lo, hi + 1)
    year = int(text)
    return range(year, year + 1)


def _run(games, epa_table, elo_table, test_season, label, kind):
    df, cols = build_features(games, epa_table=epa_table, elo_table=elo_table)
    train_df, test_df = model.time_split(df, test_season)
    pipe = model.train(train_df, cols, kind=kind)
    result = model.evaluate(pipe, test_df, cols)
    print(f"\n[{label}]  ({len(cols)} features, {kind})")
    print(result)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="NFL moneyline win-probability model")
    ap.add_argument("--train", default="2010-2022")
    ap.add_argument("--test", default="2023")
    ap.add_argument("--no-epa", action="store_true", help="skip play-by-play EPA")
    args = ap.parse_args()

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

    print("\n=== Ablation (test season "
          f"{test_season}, benchmarked vs market) ===")
    _run(games, None, None, test_season, "base form", "logistic")
    _run(games, None, elo_table, test_season, "base + Elo", "logistic")
    if epa_table is not None:
        _run(games, epa_table, elo_table, test_season, "base + Elo + EPA", "logistic")
        _run(games, epa_table, elo_table, test_season, "base + Elo + EPA", "gbm")


if __name__ == "__main__":
    main()
