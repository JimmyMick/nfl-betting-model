"""Detect which season + week the weekly preview / grader should target.

Used by the scheduled automation (Thursday preview, Tuesday grade) so the cron
job carries no hard-coded week — it asks the live schedule what's next.
"""

from __future__ import annotations

import datetime as dt

from . import data


def current_season(today: dt.date | None = None) -> int:
    """NFL season label for ``today``. A season is named by its starting year,
    and Jan/Feb belong to the prior season (playoffs), so roll back before March.
    """
    today = today or dt.date.today()
    return today.year if today.month >= 3 else today.year - 1


def detect_target(mode: str, season: int | None = None,
                  today: dt.date | None = None) -> tuple[int, int]:
    """Return ``(season, week)`` for the given ``mode``.

    ``preview`` — the earliest week that still has an unplayed game (the upcoming
    slate). ``grade`` — the most recent week with completed games.

    Raises ``SystemExit`` when there's nothing to do (off-season / season over),
    so a scheduled run can exit quietly instead of posting noise.
    """
    season = season or current_season(today)
    games = data.load_games([season], include_unplayed=True)
    if games.empty:
        raise SystemExit(f"No schedule found for {season}.")

    played = games["home_win"].notna()
    if mode == "preview":
        upcoming = games[~played]
        if upcoming.empty:
            raise SystemExit(f"{season} season is complete — nothing to preview.")
        return season, int(upcoming["week"].min())
    if mode == "grade":
        done = games[played]
        if done.empty:
            raise SystemExit(f"{season} season hasn't started — nothing to grade.")
        return season, int(done["week"].max())
    raise SystemExit(f"Unknown mode {mode!r} (use 'preview' or 'grade').")
