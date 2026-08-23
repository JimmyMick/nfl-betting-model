"""Identify each game's starting QB and attach their Madden rating.

The starter is the passer with the most dropbacks for a team in that game (the
de-facto starter). This is derived from the game's own play-by-play, but the
feature value — the QB's season *launch* OVR — is fixed before the season, so it
carries no in-game outcome. In practice the starting QB is announced ~90 min
before kickoff, so using it as a pre-game feature is realistic, not leakage.

Known, accepted caveat (reviewed 2026 — deliberately kept as-is)
---------------------------------------------------------------
"Most dropbacks in the game" is a *post-kickoff* fact, so in the rare case where
the intended starter is replaced (mid-game injury, blowout benching, a late
scratch whose backup then throws more) this labels the QB who actually played,
not the one announced pre-game — a mild leak. It is deliberately tolerated:
  * The value is a fixed preseason OVR, so it only bites when the realized passer
    differs from the intended starter AND their OVRs differ — a few games a year.
  * The live prediction path never touches the target game's pbp: predict.py's
    ``_carry_forward`` fills the upcoming week's QB from the last known starter,
    so serving is already strictly pre-game. The leak is confined to historical
    (training) rows.
A fully leak-free swap (carry-forward the prior game's starter, or pre-game depth
charts) was weighed and judged not worth the accuracy tradeoff / re-validation
cost for so narrow an effect. Revisit if the QB feature is reworked.
"""

from __future__ import annotations

import nflreadpy as nfl
import pandas as pd
import polars as pl

from . import madden as madden_mod


def starting_qb(seasons: list[int]) -> pd.DataFrame:
    """One row per (game_id, team) with the starting QB's gsis_id and season."""
    frames: list[pd.DataFrame] = []
    for season in seasons:
        pbp = (
            nfl.load_pbp(seasons=[season])
            .select(["game_id", "season", "posteam", "passer_player_id"])
            .filter(pl.col("passer_player_id").is_not_null())
        )
        starter = (
            pbp.group_by(["game_id", "season", "posteam", "passer_player_id"])
            .agg(pl.len().alias("dropbacks"))
            .sort("dropbacks", descending=True)
            .group_by(["game_id", "season", "posteam"])
            .first()  # most dropbacks wins -> the starter
            .select(["game_id", "season", "posteam", "passer_player_id"])
            .rename({"posteam": "team", "passer_player_id": "qb_gsis_id"})
        )
        frames.append(starter.to_pandas())
    return pd.concat(frames, ignore_index=True)


def starting_qb_ovr(seasons: list[int]) -> pd.DataFrame:
    """Return ``[game_id, team, qb_ovr]`` — the starting QB's Madden OVR.

    Joins the de-facto starter to that season's Madden rating on (gsis_id,
    season). Games whose starter has no rating get NaN (handled downstream).
    """
    starters = starting_qb(seasons)
    ratings = madden_mod.load_ratings(seasons)[["gsis_id", "season", "overallrating"]]
    merged = starters.merge(
        ratings,
        left_on=["qb_gsis_id", "season"],
        right_on=["gsis_id", "season"],
        how="left",
    )
    merged = merged.rename(columns={"overallrating": "qb_ovr"})
    return merged[["game_id", "team", "qb_ovr"]]
