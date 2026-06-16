"""Load Madden (EA Sports) player ratings, keyed by gsis_id.

Source: github.com/theedgepredictor/nfl-madden-data — per-season parquet files
where ``player_id`` is the nflverse gsis_id, so it joins cleanly to rosters and
play-by-play. Files are cached locally under ``data/madden/``.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

_BASE = (
    "https://raw.githubusercontent.com/theedgepredictor/nfl-madden-data/"
    "main/data/madden/dataset/{season}.parquet"
)
_CACHE = Path(__file__).parent.parent / "data" / "madden"

# Columns we keep: identity + headline rating + a few model-relevant attributes.
_KEEP = [
    "player_id", "pfr_id", "fullname", "position", "position_group", "team",
    "season", "overallrating", "speed", "acceleration", "awareness", "strength",
    "throwpower", "throwaccuracymid", "passblocking", "runblocking",
    "mancoverage", "zonecoverage",
]


def ratings_by_pfr(seasons: list[int]) -> pd.DataFrame:
    """Madden ratings keyed by pfr_id + season (for joining to snap counts)."""
    df = load_ratings(seasons)
    df = df[df["pfr_id"].notna()]
    return df.drop_duplicates(["pfr_id", "season"]).reset_index(drop=True)


def _cached_parquet(season: int) -> Path:
    _CACHE.mkdir(parents=True, exist_ok=True)
    path = _CACHE / f"{season}.parquet"
    if not path.exists():
        urllib.request.urlretrieve(_BASE.format(season=season), path)
    return path


def load_ratings(seasons: list[int]) -> pd.DataFrame:
    """Return Madden ratings for the given seasons, one row per (player, season).

    Rows without a gsis_id are dropped (can't join them to anything).
    """
    frames = []
    for season in seasons:
        df = pd.read_parquet(_cached_parquet(season))
        cols = [c for c in _KEEP if c in df.columns]
        frames.append(df[cols])
    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"player_id": "gsis_id"})
    out = out[out["gsis_id"].notna()].reset_index(drop=True)
    # One rating per (gsis_id, season): keep the highest OVR if duplicated.
    out = (
        out.sort_values("overallrating", ascending=False)
        .drop_duplicates(["gsis_id", "season"])
        .reset_index(drop=True)
    )
    out["season"] = out["season"].astype(int)
    return out


def rating_rows(seasons: list[int]) -> list[dict]:
    """Ratings as graph-ingest-ready dicts (gsis_id, season, overallrating, ...)."""
    return load_ratings(seasons).to_dict("records")
