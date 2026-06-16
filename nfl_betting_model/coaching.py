"""Coaching features from the schedule: career win% and a new-regime flag.

Both signals are strictly pre-game (leak-free):

    coach_winpct — the head coach's career win rate *entering* the game, an
        expanding mean over all their prior games (any team/season), ``shift(1)``'d
        so the current game is excluded. NaN on a coach's debut.
    coach_new    — 1.0 if this is the coach's first season with this team, the
        "new regime" discontinuity that the team's own on-field metrics (Elo,
        EPA, form) structurally cannot capture: last year's ratings describe a
        team the new staff has only partly inherited.

This is the first model signal to lean on the coaching relationships already in
the Neo4j graph, derived here straight from the schedule's ``home_coach`` /
``away_coach`` columns so it needs no extra data source.

Caveat: the new-regime flag is relative to the *loaded* season window — a coach
employed continuously from before the earliest loaded season looks "new" in that
first season only. Train from a season or two before your test window (as the
ablation does) and the flag is correct for every evaluated season.

NULL RESULT (ablation, train 2010-2022 / test 2023, 98.5% coverage): these
features add **no marginal signal** on top of team strength. Dropped onto
base+Elo+EPA the logistic model is unchanged to three decimals (logloss .655,
brier .231); on the full player set logloss/brier/AUC are identical (.631/.220/
.694) for +1 game of accuracy out of 285 (noise), and the GBM gets slightly
*worse*. Career win% is a relabeling of team quality Elo/EPA already capture, and
the new-regime flag didn't add anything orthogonal. Kept here, gated and out of
the live preview path, as a documented dead-end so it isn't re-tried from scratch
(same spirit as the betting-ROI null result). Run via ``main.py`` (default on;
``--no-coaching`` to skip).
"""

from __future__ import annotations

import nflreadpy as nfl
import pandas as pd


def _coach_long(seasons: list[int]) -> pd.DataFrame:
    """One row per (game, team) carrying that team's coach and game result.

    Unplayed games are kept with ``won = NaN`` so an upcoming slate still gets a
    pre-game career win rate; their own (missing) result never feeds the
    expanding mean.
    """
    raw = nfl.load_schedules(seasons=list(seasons))
    df = raw.to_pandas() if hasattr(raw, "to_pandas") else pd.DataFrame(raw)
    df = df[
        ["game_id", "season", "week", "gameday", "home_team", "away_team",
         "home_score", "away_score", "home_coach", "away_coach"]
    ].copy()
    df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce")

    played = df["home_score"].notna() & df["away_score"].notna()
    home_win = pd.Series(float("nan"), index=df.index)
    home_win.loc[played] = (
        df.loc[played, "home_score"] > df.loc[played, "away_score"]
    ).astype(float)

    home = pd.DataFrame({
        "game_id": df["game_id"], "season": df["season"], "gameday": df["gameday"],
        "team": df["home_team"], "coach": df["home_coach"], "won": home_win,
    })
    away = pd.DataFrame({
        "game_id": df["game_id"], "season": df["season"], "gameday": df["gameday"],
        "team": df["away_team"], "coach": df["away_coach"], "won": 1 - home_win,
    })
    long = pd.concat([home, away], ignore_index=True)
    return long.sort_values(["gameday", "game_id"]).reset_index(drop=True)


def coach_features(seasons: list[int]) -> pd.DataFrame:
    """Return ``[game_id, team, coach_winpct, coach_new]`` for the given seasons."""
    long = _coach_long(seasons)

    # Career win rate entering the game: expanding mean over the coach's prior
    # games (across teams and seasons), shifted so the current game is excluded.
    long["coach_winpct"] = long.groupby("coach", group_keys=False)["won"].apply(
        lambda s: s.shift(1).expanding().mean()
    )

    # New regime: the coach's first season with this team (within the window).
    first_season = long.groupby(["coach", "team"])["season"].transform("min")
    long["coach_new"] = (long["season"] == first_season).astype(float)

    return long[["game_id", "team", "coach_winpct", "coach_new"]]
