"""Load Madden (EA Sports) player ratings, keyed by gsis_id.

Source: github.com/theedgepredictor/nfl-madden-data — per-season parquet files
where ``player_id`` is the nflverse gsis_id, so it joins cleanly to rosters and
play-by-play. Files are cached locally under ``data/madden/``.

Download hardening: the fetch is validated (parses as parquet + has the expected
schema + non-empty) into a temp file and only *then* atomically moved into the
cache, so an interrupted download or a wrong/injected upstream file can't poison
the cache with a silently-broken parquet. The upstream ref defaults to ``main``
but can be pinned to a reviewed commit SHA via ``MADDEN_DATA_REF`` for
reproducible / supply-chain-stable builds.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import pandas as pd

# Pin to a commit SHA via MADDEN_DATA_REF for reproducibility; defaults to main.
_DATA_REF = os.getenv("MADDEN_DATA_REF", "main")
_BASE = (
    "https://raw.githubusercontent.com/theedgepredictor/nfl-madden-data/"
    "{ref}/data/madden/dataset/{season}.parquet"
)
_CACHE = Path(__file__).parent.parent / "data" / "madden"

# Minimal schema a real Madden season file must have, used to reject a wrong or
# corrupted download before it's cached.
_REQUIRED_COLS = {"player_id", "overallrating"}

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


def _validate_parquet(path: Path, season: int) -> None:
    """Reject a download that isn't a readable Madden-shaped parquet."""
    try:
        df = pd.read_parquet(path)
    except Exception as e:  # truncated / not-parquet / HTML error page etc.
        raise RuntimeError(
            f"Madden {season}: downloaded file is not readable parquet ({e})"
        ) from e
    if df.empty or not _REQUIRED_COLS.issubset(df.columns):
        raise RuntimeError(
            f"Madden {season}: unexpected/empty schema "
            f"(cols={list(df.columns)[:8]}…) — refusing to cache."
        )


def _cached_parquet(season: int) -> Path:
    _CACHE.mkdir(parents=True, exist_ok=True)
    path = _CACHE / f"{season}.parquet"
    if path.exists():
        return path
    # Download to a temp file, validate, then move into place — so a partial or
    # bad download never leaves a broken parquet that later reads treat as cached.
    tmp = path.with_name(f"{season}.parquet.tmp")
    try:
        urllib.request.urlretrieve(
            _BASE.format(ref=_DATA_REF, season=season), tmp)
        _validate_parquet(tmp, season)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
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
