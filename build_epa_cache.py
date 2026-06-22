"""Resumable, single-pass pbp -> EPA cache for the opponent-adjusted early-down
study. For each season, load play-by-play ONCE and write a small parquet with
both EPA flavours:

  off_epa / def_epa        — raw all-down means (matches epa.team_game_epa)
  off_raw_ed / def_raw_ed  — raw early-down (downs 1-2) means

Caches to data/epa_cache/{season}.parquet and skips seasons already cached, so
repeated runs make progress even if the process is killed mid-load.

Run (repeatedly if needed):  uv run python build_epa_cache.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import nflreadpy as nfl
import pandas as pd
import polars as pl

CACHE = Path("data/epa_cache")
CACHE.mkdir(parents=True, exist_ok=True)
SEASONS = list(range(2010, 2026))


def _means(pbp: pl.DataFrame, off_name: str, def_name: str) -> pd.DataFrame:
    off = (pbp.filter(pl.col("posteam").is_not_null())
           .group_by(["game_id", "posteam"]).agg(pl.col("epa").mean().alias(off_name))
           .rename({"posteam": "team"}))
    deff = (pbp.filter(pl.col("defteam").is_not_null())
            .group_by(["game_id", "defteam"]).agg(pl.col("epa").mean().alias(def_name))
            .rename({"defteam": "team"}))
    return off.join(deff, on=["game_id", "team"], how="full", coalesce=True).to_pandas()


def build_season(season: int) -> None:
    pbp = (nfl.load_pbp(seasons=[season])
           .select(["game_id", "posteam", "defteam", "epa", "down"])
           .filter(pl.col("epa").is_not_null()))
    alld = _means(pbp, "off_epa", "def_epa")
    ed = _means(pbp.filter(pl.col("down").is_in([1, 2])), "off_raw_ed", "def_raw_ed")
    out = alld.merge(ed, on=["game_id", "team"], how="outer")
    out.to_parquet(CACHE / f"{season}.parquet")
    print(f"  cached {season}: {len(out)} team-games", flush=True)


done = 0
for s in SEASONS:
    fp = CACHE / f"{s}.parquet"
    if fp.exists():
        continue
    print(f"loading {s} ...", flush=True)
    build_season(s)
    done += 1

missing = [s for s in SEASONS if not (CACHE / f"{s}.parquet").exists()]
print(f"\nbuilt {done} new | cached {len(SEASONS) - len(missing)}/{len(SEASONS)} "
      f"| missing {missing}", flush=True)
sys.exit(0 if not missing else 1)
