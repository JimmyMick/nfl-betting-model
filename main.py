"""Train and evaluate the NFL moneyline win-probability model.

Examples
--------
    uv run main.py --train 2010-2022 --test 2023
"""

from __future__ import annotations

import argparse

from nfl_betting_model import data, features, model


def _parse_range(text: str) -> range:
    """Parse '2010-2022' or a single '2023' into an inclusive range."""
    if "-" in text:
        lo, hi = (int(x) for x in text.split("-", 1))
        return range(lo, hi + 1)
    year = int(text)
    return range(year, year + 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="NFL moneyline win-probability model")
    ap.add_argument("--train", default="2010-2022", help="train seasons, e.g. 2010-2022")
    ap.add_argument("--test", default="2023", help="test season, e.g. 2023")
    args = ap.parse_args()

    train_seasons = _parse_range(args.train)
    test_season = int(args.test)

    all_seasons = list(train_seasons) + [test_season]
    print(f"Loading schedules for {all_seasons[0]}-{all_seasons[-1]} ...")
    games = data.load_games(all_seasons)
    print(f"  {len(games)} completed games loaded")

    df = features.build_features(games)
    print(f"  {len(df)} games with usable pre-game features")

    train_df, test_df = model.time_split(df, test_season)
    print(f"  train={len(train_df)}  test={len(test_df)}\n")

    pipe = model.train(train_df)
    result = model.evaluate(pipe, test_df)
    print(result)


if __name__ == "__main__":
    main()
