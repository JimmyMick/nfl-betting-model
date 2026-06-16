"""Per-team season roster: who played, how much, and how Madden rates them.

Built from nflverse snap counts (authoritative for who actually played and their
snap share) joined to Madden ratings on ``pfr_id``. A player is flagged a
*starter* when their season-average snap share on offense or defense clears
``_SNAP_THRESHOLD`` — the same 50% rule the model's starter features use.

Snap counts begin in 2012 and only exist for games already played, so this view
needs a season that's underway; an upcoming season (no snaps yet) raises
``SystemExit``.
"""

from __future__ import annotations

import nflreadpy as nfl
import pandas as pd

from . import madden as madden_mod

_SNAP_THRESHOLD = 0.5

# Map snap-count position codes to a side of the ball, for grouping/ordering.
_OFFENSE = {"QB", "RB", "FB", "HB", "WR", "TE", "C", "G", "T", "OL",
            "LT", "RT", "LG", "RG", "OT", "OG"}
_DEFENSE = {"DE", "DT", "NT", "DL", "EDGE", "LB", "ILB", "OLB", "MLB",
            "CB", "DB", "S", "FS", "SS", "SAF"}
_SPECIAL = {"K", "P", "LS", "PK"}

_UNIT_ORDER = {"Offense": 0, "Defense": 1, "Special teams": 2, "Other": 3}

# Madden attributes worth showing next to the headline rating.
_ATTRS = ["speed", "acceleration", "awareness", "strength"]


def _unit_of(position: str) -> str:
    if position in _OFFENSE:
        return "Offense"
    if position in _DEFENSE:
        return "Defense"
    if position in _SPECIAL:
        return "Special teams"
    return "Other"


def team_roster(team: str, season: int) -> pd.DataFrame:
    """Return one row per player on ``team`` in ``season``.

    Columns: ``player, position, unit, games, snap_share, starter,
    overallrating`` plus the attributes in ``_ATTRS``. Sorted by unit then snap
    share (most-used first), so starters surface at the top of each unit.
    """
    sc = nfl.load_snap_counts(seasons=[season]).to_pandas()
    sc = sc[(sc["team"] == team) & sc["pfr_player_id"].notna()].copy()
    if sc.empty:
        raise SystemExit(
            f"No snap-count data for {team} in {season} "
            "(needs a season that's already underway)."
        )

    # Each player's most-played position across the season.
    pos = (
        sc.groupby(["pfr_player_id", "position"]).size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
        .drop_duplicates("pfr_player_id")[["pfr_player_id", "position"]]
    )

    agg = (
        sc.groupby(["pfr_player_id", "player"]).agg(
            games=("game_id", "nunique"),
            off_share=("offense_pct", "mean"),
            def_share=("defense_pct", "mean"),
            st_share=("st_pct", "mean"),
        ).reset_index()
        .merge(pos, on="pfr_player_id", how="left")
    )

    ratings = madden_mod.ratings_by_pfr([season])
    keep = ["pfr_id", "overallrating"] + [a for a in _ATTRS if a in ratings.columns]
    agg = agg.merge(ratings[keep], left_on="pfr_player_id", right_on="pfr_id",
                    how="left")

    agg["snap_share"] = agg[["off_share", "def_share"]].max(axis=1)
    agg["starter"] = agg["snap_share"] >= _SNAP_THRESHOLD
    agg["unit"] = agg["position"].map(_unit_of)
    agg["_unit_order"] = agg["unit"].map(_UNIT_ORDER)

    cols = (["player", "position", "unit", "games", "snap_share", "starter",
             "overallrating"] + [a for a in _ATTRS if a in agg.columns])
    return (
        agg.sort_values(["_unit_order", "snap_share"], ascending=[True, False])
        [cols].reset_index(drop=True)
    )
