"""Monte Carlo the rest of the NFL season and (optionally) snapshot the odds.

Rolls the remaining schedule forward thousands of times using the win-probability
model, then reports each team's projected wins and playoff / division / #1-seed /
conference / Super Bowl odds. With --export-dir it writes the cloud artifacts
(current snapshot + appended weekly history) so both apps can render it and the
weekly cron persists it — giving a week-to-week record to compare against actual.

    ./.venv/bin/python simulate_season.py                    # print current odds
    ./.venv/bin/python simulate_season.py --export-dir       # + save snapshot
    ./.venv/bin/python simulate_season.py --season 2026 --sims 20000
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from nfl_betting_model import cloud, data, simulate as sim


def _default_season() -> int:
    today = dt.date.today()
    return today.year - 1 if today.month <= 2 else today.year


def _season_state(season: int) -> tuple[int, int]:
    """Return ``(through_week, n_remaining)`` for this season's regular season."""
    g = data.load_games([season], include_unplayed=True)
    reg = g[(g["season"] == season) & (g["game_type"] == "REG")]
    played = reg[reg["home_win"].notna()]
    through = int(played["week"].max()) if len(played) else 0
    return through, int(reg["home_win"].isna().sum())


def render(proj, season: int, through_week: int) -> str:
    when = "preseason" if through_week == 0 else f"through Week {through_week}"
    lines = [f"# NFL playoff odds — {season} ({when})", "",
             "_Monte Carlo of the rest of the season using the model. "
             "Just for fun — not a betting product._", "",
             "| Team | Conf | Proj W | Playoffs | Division | #1 Seed | Conf | Super Bowl |",
             "|---|---|--:|--:|--:|--:|--:|--:|"]
    for _, r in proj.head(16).iterrows():
        lines.append(
            f"| {r['team']} | {r['conference']} | {r['proj_wins']:.1f} | "
            f"{r['make_playoffs']:.0%} | {r['win_division']:.0%} | "
            f"{r['top_seed']:.0%} | {r['win_conference']:.0%} | {r['win_sb']:.0%} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Monte Carlo NFL season simulator")
    ap.add_argument("--season", type=int, default=_default_season())
    ap.add_argument("--sims", type=int, default=10000)
    ap.add_argument("--export-dir", nargs="?", const="", default=None,
                    help="write cloud artifacts here (bare = predictions/cloud)")
    ap.add_argument("--out", type=str, help="also write the markdown table here")
    args = ap.parse_args()

    through_week, n_remaining = _season_state(args.season)
    if n_remaining == 0:
        print(f"{args.season} regular season complete — nothing to simulate.")
        return

    proj = sim.simulate(args.season, n=args.sims)
    report = render(proj, args.season, through_week)
    print("\n" + report)

    if args.out:
        Path(args.out).write_text(report + "\n")
        print(f"\nWrote {args.out}")

    if args.export_dir is not None:
        out_dir = Path(args.export_dir) if args.export_dir else cloud.ARTIFACT_DIR
        cloud.write_sim_artifacts(proj, args.season, through_week, out_dir)
        print(f"Exported playoff-odds artifacts to {out_dir}")


if __name__ == "__main__":
    main()
