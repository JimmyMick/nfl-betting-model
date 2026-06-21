"""Seed weekly pick sheets for the expert pick'em tracker.

Writes a blank ``predictions/picks/{season}-wk{week:02d}.csv`` — one row per
(game, player) drawn from that week's schedule, with empty ``pick`` and
``confidence`` columns to fill in before kickoff. Players come from
``predictions/picks/players.txt`` (one name per line).

Scoring lives in grade.py (the picks leaderboard rides along with the Tuesday
model grade); this CLI is just the collection half.

Examples
--------
    uv run picks.py --auto                 # seed the upcoming slate
    uv run picks.py --season 2026 --week 1
    uv run picks.py --season 2026 --week 1 --force   # overwrite existing sheet
"""

from __future__ import annotations

import argparse

from nfl_betting_model import data, picks


def main() -> None:
    ap = argparse.ArgumentParser(description="Seed a weekly pick sheet")
    ap.add_argument("--season", type=int, help="omit with --auto")
    ap.add_argument("--week", type=int, help="omit with --auto")
    ap.add_argument("--auto", action="store_true",
                    help="detect the upcoming slate from the live schedule")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing sheet (erases filled picks!)")
    args = ap.parse_args()

    if args.auto:
        from nfl_betting_model.weeks import detect_target
        season, week = detect_target("preview", args.season)
    elif args.season is not None and args.week is not None:
        season, week = args.season, args.week
    else:
        ap.error("provide --season and --week, or --auto")

    players = picks.load_players()
    if not players:
        raise SystemExit(
            "No players found. Add one name per line to "
            f"{picks.PLAYERS_FILE.relative_to(picks.PICKS_DIR.parent.parent)}."
        )

    games = data.load_games([season], include_unplayed=True)
    gw = games[(games["season"] == season) & (games["week"] == week)]
    if gw.empty:
        raise SystemExit(f"No games found for {season} week {week}.")

    path, written = picks.seed_week(gw, players, picks.week_path(season, week),
                                    force=args.force)
    if not written:
        raise SystemExit(
            f"{path} already exists — not overwriting. Use --force to replace it."
        )
    print(f"Seeded {len(gw)} games x {len(players)} players -> {path}")
    print(f"Players: {', '.join(players)}")
    print("Fill in the 'pick' (team abbrev) and 'confidence' (50-100) columns.")


if __name__ == "__main__":
    main()
