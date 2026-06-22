"""Opponent-adjusted, early-down EPA per team-game (experimental).

A richer EPA flavour than ``epa.py``'s raw all-down means, testing two ideas at
once (see ``validate_epa_splits.py``):

* **Early-down** — restrict to 1st & 2nd down (``down in {1, 2}``), the snaps
  most analysts consider the "stickiest"/least game-script-contaminated.
* **Opponent-adjusted** — credit a team for the strength of the defense/offense
  it faced. The adjustment uses only each opponent's **prior** games in the
  season (an expanding ``shift(1)`` baseline), so it carries no information from
  the game being rated — leak-free in the same sense as the rolling form
  features. The per-game adjusted value is still rolled again by the feature
  layer before use.

Output columns are ``off_epa`` / ``def_epa`` so the table is a drop-in
replacement for ``epa.team_game_epa`` in ``build_features(epa_table=...)``.
"""

from __future__ import annotations

import nflreadpy as nfl
import pandas as pd
import polars as pl


def _raw_early_down(seasons: list[int]) -> pd.DataFrame:
    """Per (game_id, team) raw early-down off/def EPA means."""
    frames: list[pd.DataFrame] = []
    for season in seasons:
        pbp = (
            nfl.load_pbp(seasons=[season])
            .select(["game_id", "posteam", "defteam", "epa", "down"])
            .filter(pl.col("epa").is_not_null())
            .filter(pl.col("down").is_in([1, 2]))
        )
        off = (
            pbp.filter(pl.col("posteam").is_not_null())
            .group_by(["game_id", "posteam"])
            .agg(pl.col("epa").mean().alias("off_raw"))
            .rename({"posteam": "team"})
        )
        deff = (
            pbp.filter(pl.col("defteam").is_not_null())
            .group_by(["game_id", "defteam"])
            .agg(pl.col("epa").mean().alias("def_raw"))
            .rename({"defteam": "team"})
        )
        merged = off.join(deff, on=["game_id", "team"], how="full", coalesce=True)
        frames.append(merged.to_pandas())
    return pd.concat(frames, ignore_index=True)


def team_game_epa_oa_ed(seasons: list[int], games: pd.DataFrame) -> pd.DataFrame:
    """Opponent-adjusted early-down EPA per (game_id, team).

    ``games`` supplies chronological order (``gameday``), season, and each
    team's opponent — needed to compute the leak-free opponent baselines.
    """
    return adjust_early_down(_raw_early_down(seasons), games)


def adjust_early_down(raw: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Opponent-adjust a raw early-down EPA frame (``game_id, team, off_raw,
    def_raw``). Split out so a cached raw frame can be reused without re-loading
    play-by-play (see ``build_epa_cache.py`` / ``validate_epa_splits.py``).
    """
    # Attach schedule context: one row per (game_id, team) with opponent/date.
    sched = pd.concat([
        games[["game_id", "gameday", "season", "home_team", "away_team"]].rename(
            columns={"home_team": "team", "away_team": "opponent"}),
        games[["game_id", "gameday", "season", "away_team", "home_team"]].rename(
            columns={"away_team": "team", "home_team": "opponent"}),
    ], ignore_index=True)

    df = sched.merge(raw, on=["game_id", "team"], how="left")
    df = df.sort_values(["team", "gameday"]).reset_index(drop=True)

    # Each team's pre-game baseline: expanding mean of its own prior games this
    # season (shift(1) => excludes the current game). League EPA ~ 0, so unknown
    # baselines (a team's season opener) fall back to 0.
    def _prior(col: str) -> pd.Series:
        return df.groupby(["team", "season"], group_keys=False)[col].apply(
            lambda s: s.shift(1).expanding().mean()).fillna(0.0)

    df["off_base"] = _prior("off_raw")
    df["def_base"] = _prior("def_raw")

    # Look up the opponent's pre-game baselines for the same game.
    base = df.set_index(["game_id", "team"])[["off_base", "def_base"]]
    opp_idx = pd.MultiIndex.from_arrays([df["game_id"], df["opponent"]])
    opp_off_base = base["off_base"].reindex(opp_idx).to_numpy()
    opp_def_base = base["def_base"].reindex(opp_idx).to_numpy()

    # Adjust: credit offense for facing a stingy D, defense for facing a strong O.
    df["off_epa"] = df["off_raw"] - opp_def_base
    df["def_epa"] = df["def_raw"] - opp_off_base

    return df[["game_id", "team", "off_epa", "def_epa"]].dropna(
        subset=["off_epa", "def_epa"], how="all").reset_index(drop=True)
