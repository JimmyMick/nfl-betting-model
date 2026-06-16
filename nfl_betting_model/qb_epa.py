"""Per-player rolling QB EPA/play, attributed to each game's starter.

Unlike team ``off_epa`` (epa.py), this travels with the *quarterback*: each
game's value is the starter's mean passing EPA/play over his own PRIOR games
(``shift(1)`` rolling, ordered by date across teams and seasons). So a backup
making his first start carries his own thin history (not the team's), and a QB
who changed teams brings his form with him.

Leakage: identical discipline to team EPA — the current game is excluded
(``shift(1)``), and the starter is the most-dropbacks passer (announced ~90 min
pre-kickoff, so realistic as a pre-game feature). The window spans seasons on
purpose: a QB's recent form is his recent form regardless of the calendar.

NULL RESULT (ablation, train 2010-2022 / test 2023, 1076/1138 starters with
prior EPA): adds **no marginal signal**. Dropped onto base+Elo+EPA the logistic
model is unchanged to three decimals (logloss .655, brier .231, AUC .651) with
*lower* accuracy, and the GBM gets worse (.660->.664 logloss). Layered on the
Madden QB OVR it's flat-to-worse on calibration and AUC drops (.672->.669); the
lone logistic accuracy bump (.646->.660) isn't backed by calibration or ranking,
i.e. noise. Root cause: in the modern pass-heavy NFL the QB drives most of team
offensive EPA, so the team ``off_epa`` we already roll *is* mostly the QB — this
is redundant with it, and it doesn't complement the static preseason QB rating.
Kept gated and out of the live preview path as a documented dead-end (cf. the
coaching and betting-ROI null results). Run via ``main.py`` (default on;
``--no-qb-epa`` to skip).
"""

from __future__ import annotations

import nflreadpy as nfl
import pandas as pd
import polars as pl

from .qb import starting_qb

# Rolling window (number of the QB's prior games) for his EPA/play form.
# Wider than the 5-game team-form window: per-QB EPA is noisier game-to-game.
QB_EPA_WINDOW = 10


def qb_game_epa(seasons: list[int]) -> pd.DataFrame:
    """One row per (game, passer): that QB's mean passing EPA/play in the game.

    Returns ``[game_id, gameday, qb_gsis_id, epa_pp]``. Aggregates pbp one
    season at a time to bound memory.
    """
    frames: list[pd.DataFrame] = []
    for season in seasons:
        pbp = (
            nfl.load_pbp(seasons=[season])
            .select(["game_id", "game_date", "passer_player_id", "epa"])
            .filter(pl.col("passer_player_id").is_not_null())
            .filter(pl.col("epa").is_not_null())
        )
        per_game = (
            pbp.group_by(["game_id", "game_date", "passer_player_id"])
            .agg(pl.col("epa").mean().alias("epa_pp"))
            .rename({"passer_player_id": "qb_gsis_id"})
        )
        frames.append(per_game.to_pandas())

    out = pd.concat(frames, ignore_index=True)
    out["gameday"] = pd.to_datetime(out["game_date"], errors="coerce")
    return out[["game_id", "gameday", "qb_gsis_id", "epa_pp"]]


def starting_qb_epa(seasons: list[int]) -> pd.DataFrame:
    """Return ``[game_id, team, qb_epa]`` — the starter's pre-game rolling EPA.

    ``qb_epa`` is the starting QB's mean passing EPA/play over his prior
    ``QB_EPA_WINDOW`` games (excluding the current one). NaN for a QB's debut.
    """
    qge = qb_game_epa(seasons)

    # Each QB's prior-games rolling EPA/play, ordered chronologically. game_id is
    # a deterministic tiebreaker for games sharing a date.
    qge = qge.sort_values(["qb_gsis_id", "gameday", "game_id"]).reset_index(drop=True)
    qge["qb_epa"] = qge.groupby("qb_gsis_id", group_keys=False)["epa_pp"].apply(
        lambda s: s.shift(1).rolling(QB_EPA_WINDOW, min_periods=1).mean()
    )

    # Attach each (game, team)'s starter to his own rolling value for that game.
    starters = starting_qb(seasons)  # [game_id, season, team, qb_gsis_id]
    merged = starters.merge(
        qge[["game_id", "qb_gsis_id", "qb_epa"]],
        on=["game_id", "qb_gsis_id"],
        how="left",
    )
    return merged[["game_id", "team", "qb_epa"]]
