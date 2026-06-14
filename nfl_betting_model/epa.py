"""Per-team, per-game EPA (expected points added) efficiency from play-by-play.

Offense EPA/play = mean EPA on plays the team ran.
Defense EPA/play = mean EPA the team *allowed* (plays where it was on defense).
Lower defensive EPA is better. These are raw game outcomes; the feature layer
turns them into leak-free rolling form.
"""

from __future__ import annotations

import nflreadpy as nfl
import pandas as pd
import polars as pl


def team_game_epa(seasons: list[int]) -> pd.DataFrame:
    """Return one row per (game_id, team) with ``off_epa`` and ``def_epa``.

    Loads play-by-play one season at a time and aggregates immediately so peak
    memory stays bounded despite pbp's ~370 columns.
    """
    frames: list[pd.DataFrame] = []
    for season in seasons:
        pbp = (
            nfl.load_pbp(seasons=[season])
            .select(["game_id", "posteam", "defteam", "epa"])
            .filter(pl.col("epa").is_not_null())
        )

        off = (
            pbp.filter(pl.col("posteam").is_not_null())
            .group_by(["game_id", "posteam"])
            .agg(pl.col("epa").mean().alias("off_epa"))
            .rename({"posteam": "team"})
        )
        deff = (
            pbp.filter(pl.col("defteam").is_not_null())
            .group_by(["game_id", "defteam"])
            .agg(pl.col("epa").mean().alias("def_epa"))
            .rename({"defteam": "team"})
        )

        merged = off.join(deff, on=["game_id", "team"], how="full", coalesce=True)
        frames.append(merged.to_pandas())

    return pd.concat(frames, ignore_index=True)
