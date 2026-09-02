"""Export the current-season matchup schedule artifact for the cloud dashboard.

Dependency-light: fetches only the season schedule (no play-by-play, Madden, or
training), writes ``predictions/cloud/schedule.csv`` + meta, and is safe to run
any time — including the off-season, so the Schedule tab has content before the
weekly preview/grade runs start. The weekly runs also refresh this artifact so
final scores fill in as games are played.

    ./.venv/bin/python export_schedule.py            # current season -> predictions/cloud
    ./.venv/bin/python export_schedule.py --season 2026 --export-dir some/dir
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from nfl_betting_model import cloud, data


def _default_season() -> int:
    # NFL seasons are labelled by their starting year; treat Jan/Feb as the
    # prior year's postseason so the off-season still targets the right slate.
    today = dt.date.today()
    return today.year - 1 if today.month <= 2 else today.year


def main() -> None:
    ap = argparse.ArgumentParser(description="Export season schedule artifact")
    ap.add_argument("--season", type=int, default=_default_season())
    ap.add_argument("--export-dir", nargs="?", const="", default="",
                    help="output dir (bare/empty = predictions/cloud)")
    args = ap.parse_args()

    games = data.load_games([args.season], include_unplayed=True)
    out_dir = Path(args.export_dir) if args.export_dir else cloud.ARTIFACT_DIR
    cloud.write_schedule_artifacts(games, args.season, out_dir)

    sched = games[games["season"] == args.season]
    reg = sched[sched.get("game_type", "REG").isin(["REG", "POST"])] \
        if "game_type" in sched.columns else sched
    print(f"Exported {len(reg)} {args.season} games to "
          f"{out_dir / cloud.SCHEDULE_FILE}")


if __name__ == "__main__":
    main()
