"""Detect which season + week the weekly preview / grader should target.

Used by the scheduled automation (Thursday preview, Tuesday grade) so the cron
job carries no hard-coded week — it asks the live schedule what's next.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from . import data

# Only preview a slate once its first game is within this many days. Keeps the
# Thursday job from posting a week early before the season opener (Week 1 opens
# 6-7 days out on the first Thursday of September) while still firing every
# in-season Thursday, when the upcoming slate is at most ~3-4 days away.
PREVIEW_HORIZON_DAYS = 5


def current_season(today: dt.date | None = None) -> int:
    """NFL season label for ``today``. A season is named by its starting year,
    and Jan/Feb belong to the prior season (playoffs), so roll back before March.
    """
    today = today or dt.date.today()
    return today.year if today.month >= 3 else today.year - 1


def detect_target(mode: str, season: int | None = None,
                  today: dt.date | None = None,
                  horizon_days: int = PREVIEW_HORIZON_DAYS) -> tuple[int, int]:
    """Return ``(season, week)`` for the given ``mode``.

    ``preview`` — the earliest unplayed week whose first game is within
    ``horizon_days`` (the imminent slate). ``grade`` — the most recent week with
    completed games.

    Raises ``SystemExit`` when there's nothing to do (off-season / season over /
    next slate still too far out), so a scheduled run can exit quietly instead of
    posting noise.
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
        wk = int(upcoming["week"].min())
        first_game = pd.to_datetime(upcoming.loc[upcoming["week"] == wk, "gameday"]).min()
        today_ts = pd.Timestamp(today or dt.date.today()).normalize()
        days_out = (first_game.normalize() - today_ts).days
        if days_out > horizon_days:
            raise SystemExit(
                f"{season} week {wk} opens {first_game.date()} ({days_out}d out) "
                f"— too early to preview."
            )
        return season, wk
    if mode == "grade":
        done = games[played]
        if done.empty:
            raise SystemExit(f"{season} season hasn't started — nothing to grade.")
        return season, int(done["week"].max())
    raise SystemExit(f"Unknown mode {mode!r} (use 'preview' or 'grade').")
