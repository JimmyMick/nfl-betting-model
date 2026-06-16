"""Per-game starting-unit talent from snap counts + Madden ratings.

Snap counts tell us who actually played (and how much); Madden tells us how good
they are. We join the two on ``pfr_id`` and aggregate the *starters* (players
above a snap-share threshold) into per-unit average overall ratings:

    ol_ovr   — starting offensive line (C/G/T)
    dl_ovr   — starting defensive line (DE/DT/NT)
    db_ovr   — starting secondary (CB/FS/SS)
    starter_ovr — all starters, both sides

These are the line/starter strength signals the team aggregates never captured.
Like the QB feature, the rating itself is fixed pre-season, so it carries no
in-game outcome.
"""

from __future__ import annotations

import nflreadpy as nfl
import pandas as pd

from . import madden as madden_mod

# Minimum snap share to count as a "starter" for a unit.
_SNAP_THRESHOLD = 0.5

# nflverse snap counts only go back to 2012; earlier seasons yield NaN features.
_SNAP_MIN_SEASON = 2012

OL_POS = {"C", "G", "T"}
DL_POS = {"DE", "DT", "NT"}
DB_POS = {"CB", "FS", "SS"}


def _snap_counts(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        if season < _SNAP_MIN_SEASON:
            continue
        sc = nfl.load_snap_counts(seasons=[season]).to_pandas()
        frames.append(
            sc[["game_id", "season", "team", "pfr_player_id", "position",
                "offense_pct", "defense_pct"]]
        )
    out = pd.concat(frames, ignore_index=True)
    return out[out["pfr_player_id"].notna()]


def starter_unit_ovr(seasons: list[int]) -> pd.DataFrame:
    """Return ``[game_id, team, ol_ovr, dl_ovr, db_ovr, starter_ovr]``."""
    sc = _snap_counts(seasons)
    ratings = madden_mod.ratings_by_pfr(seasons)[["pfr_id", "season", "overallrating"]]
    sc = sc.merge(
        ratings, left_on=["pfr_player_id", "season"],
        right_on=["pfr_id", "season"], how="left",
    )

    off_start = sc["offense_pct"] >= _SNAP_THRESHOLD
    def_start = sc["defense_pct"] >= _SNAP_THRESHOLD

    def _unit(mask: pd.Series, positions: set[str] | None) -> pd.DataFrame:
        sub = sc[mask]
        if positions is not None:
            sub = sub[sub["position"].isin(positions)]
        return sub.groupby(["game_id", "team"])["overallrating"].mean()

    out = pd.DataFrame({
        "ol_ovr": _unit(off_start, OL_POS),
        "dl_ovr": _unit(def_start, DL_POS),
        "db_ovr": _unit(def_start, DB_POS),
        "starter_ovr": _unit(off_start | def_start, None),
    }).reset_index()
    return out
