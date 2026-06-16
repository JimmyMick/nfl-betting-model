"""Load NFL game data via nflreadpy and shape it for modeling."""

from __future__ import annotations

from collections.abc import Iterable

import nflreadpy as nfl
import pandas as pd

# Schedule columns we care about (others are ignored if present).
_KEEP = [
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",
    "away_team",
    "home_team",
    "away_score",
    "home_score",
    "home_rest",
    "away_rest",
    "home_moneyline",
    "away_moneyline",
    "spread_line",
    "div_game",
    "roof",
    "location",
]


def load_games(
    seasons: Iterable[int] | None = None,
    include_unplayed: bool = False,
) -> pd.DataFrame:
    """Return NFL games as a tidy pandas frame.

    Parameters
    ----------
    seasons:
        Iterable of season years (e.g. ``range(2010, 2025)``). ``None`` loads
        every available season.
    include_unplayed:
        If ``True``, also keep scheduled games that haven't been played yet
        (``home_win`` is ``NaN`` for those). Used by the weekly inference path
        to predict an upcoming slate. Defaults to ``False`` so training callers
        get completed games only, exactly as before.
    """
    raw = nfl.load_schedules(seasons=True if seasons is None else list(seasons))

    # nflreadpy returns polars; convert to pandas for the sklearn pipeline.
    df = raw.to_pandas() if hasattr(raw, "to_pandas") else pd.DataFrame(raw)

    keep = [c for c in _KEEP if c in df.columns]
    df = df[keep].copy()

    played = df["home_score"].notna() & df["away_score"].notna()
    if not include_unplayed:
        df = df[played].copy()
        played = pd.Series(True, index=df.index)

    df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce")
    # game_id is a deterministic tiebreaker: same-day games would otherwise keep
    # nflreadpy's load order, which varies between runs and leaks into isotonic
    # calibration's tie handling (~1% probability wobble on re-run).
    df = df.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    played = df["home_score"].notna() & df["away_score"].notna()

    # Drop played ties (rare) — undefined for a binary winner model. Unplayed
    # games are kept with home_win = NaN.
    df = df[~(played & (df["home_score"] == df["away_score"]))].copy()
    played = df["home_score"].notna() & df["away_score"].notna()

    # Target: did the home team win? NaN where the game hasn't been played.
    df["home_win"] = float("nan")
    df.loc[played, "home_win"] = (
        df.loc[played, "home_score"] > df.loc[played, "away_score"]
    ).astype(float)

    return df.reset_index(drop=True)


def to_long(games: pd.DataFrame) -> pd.DataFrame:
    """Explode each game into two team-perspective rows (home + away).

    Used by the feature layer to compute rolling team form. Each row is one
    team's view of one game it played.
    """
    home = pd.DataFrame(
        {
            "game_id": games["game_id"],
            "gameday": games["gameday"],
            "season": games["season"],
            "team": games["home_team"],
            "opponent": games["away_team"],
            "is_home": 1,
            "points_for": games["home_score"],
            "points_against": games["away_score"],
            "won": games["home_win"],
        }
    )
    away = pd.DataFrame(
        {
            "game_id": games["game_id"],
            "gameday": games["gameday"],
            "season": games["season"],
            "team": games["away_team"],
            "opponent": games["home_team"],
            "is_home": 0,
            "points_for": games["away_score"],
            "points_against": games["home_score"],
            "won": 1 - games["home_win"],
        }
    )
    long = pd.concat([home, away], ignore_index=True)
    return long.sort_values(["team", "gameday"]).reset_index(drop=True)
