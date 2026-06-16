"""Starter availability from the official injury report (talent ruled out).

For each team-week this sums the *talent above replacement* of the players the
injury report lists as Out/Doubtful, valued by their Madden OVR. This is the one
signal genuinely orthogonal to team Elo/EPA: those rate a team's baseline roster
but are blind to a key starter — above all the quarterback — being ruled out
THIS week. Because QBs carry the highest OVRs, a QB ruled out dominates the sum
naturally, with no position special-casing.

    out_avail = sum over Out/Doubtful players of  weight * max(0, OVR - 65)

Leakage: the injury report's game-status designation is published days before
kickoff (Wed-Fri practice reports + the Friday final status), so it is exactly
the pre-game information a bettor holds — not leakage. We use ``report_status``
only, never actual inactives or snap counts (known at kickoff).
"""

from __future__ import annotations

import nflreadpy as nfl
import pandas as pd

from . import madden as madden_mod

# Madden OVR at/below which an out player costs ~no talent (replacement-level).
REPLACEMENT_OVR = 65
# Likelihood each game-status designation actually misses the game.
# Questionable is excluded (most such players suit up).
_STATUS_WEIGHT = {"Out": 1.0, "Doubtful": 0.75}


def _schedule_team_games(seasons: list[int]) -> pd.DataFrame:
    """[game_id, season, week, team] — one row per team per scheduled game."""
    raw = nfl.load_schedules(seasons=list(seasons))
    df = raw.to_pandas() if hasattr(raw, "to_pandas") else pd.DataFrame(raw)
    df = df[["game_id", "season", "week", "home_team", "away_team"]]
    home = df.rename(columns={"home_team": "team"})[
        ["game_id", "season", "week", "team"]
    ]
    away = df.rename(columns={"away_team": "team"})[
        ["game_id", "season", "week", "team"]
    ]
    return pd.concat([home, away], ignore_index=True)


def team_out_talent(seasons: list[int]) -> pd.DataFrame:
    """Return ``[game_id, team, out_avail]`` — summed talent-above-replacement out.

    Every scheduled team-game gets a row; teams with no rated player ruled out
    get ``0.0`` (healthy), never NaN.
    """
    # Load one season at a time and skip any without an injury feed yet (a
    # not-yet-played future season raises in nflreadpy) — those games simply get
    # the neutral 0.0 default below.
    frames: list[pd.DataFrame] = []
    for s in seasons:
        try:
            one = nfl.load_injuries(seasons=[s])
        except ValueError:
            continue
        frames.append(one.to_pandas() if hasattr(one, "to_pandas") else pd.DataFrame(one))

    cols = ["season", "week", "team", "gsis_id", "report_status"]
    inj = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
    inj = inj[inj["report_status"].isin(_STATUS_WEIGHT)][cols].copy()
    inj["weight"] = inj["report_status"].map(_STATUS_WEIGHT)

    # Only fetch Madden ratings for seasons that actually have injuries (skips a
    # not-yet-rated future season, whose parquet 404s).
    inj_seasons = sorted(int(s) for s in inj["season"].dropna().unique())
    ratings = (
        madden_mod.load_ratings(inj_seasons)[["gsis_id", "season", "overallrating"]]
        if inj_seasons
        else pd.DataFrame(columns=["gsis_id", "season", "overallrating"])
    )
    inj = inj.merge(ratings, on=["gsis_id", "season"], how="left")
    above_repl = (inj["overallrating"] - REPLACEMENT_OVR).clip(lower=0)
    inj["contribution"] = inj["weight"] * above_repl.fillna(0.0)

    per_team = (
        inj.groupby(["season", "week", "team"])["contribution"]
        .sum()
        .reset_index()
        .rename(columns={"contribution": "out_avail"})
    )

    sched = _schedule_team_games(seasons)
    out = sched.merge(per_team, on=["season", "week", "team"], how="left")
    out["out_avail"] = out["out_avail"].fillna(0.0)
    return out[["game_id", "team", "out_avail"]]
