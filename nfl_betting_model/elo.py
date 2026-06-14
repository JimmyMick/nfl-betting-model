"""538-style Elo ratings for NFL teams.

Produces strictly pre-game features: each game is rated using the ratings as
they stood *before* kickoff, then the ratings are updated from the result. No
leakage.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

BASE = 1500.0       # starting / mean rating
HFA = 55.0          # home-field advantage in Elo points
K = 20.0            # update speed
REVERT = 0.33       # fraction reverted toward BASE at the start of each season


def _expected(elo_home: float, elo_away: float) -> float:
    """Expected home win probability (HFA already folded into elo_home)."""
    return 1.0 / (1.0 + 10 ** (-(elo_home - elo_away) / 400.0))


def _mov_multiplier(margin: int, elo_diff_winner: float) -> float:
    """FiveThirtyEight margin-of-victory multiplier.

    Scales the update by how lopsided the result was, while damping the
    autocorrelation that favors already-strong teams (the elo_diff term).
    """
    return math.log(abs(margin) + 1.0) * (2.2 / (elo_diff_winner * 0.001 + 2.2))


def compute_elo(
    games: pd.DataFrame,
    *,
    base: float = BASE,
    hfa: float = HFA,
    k: float = K,
    revert: float = REVERT,
) -> pd.DataFrame:
    """Return per-game pre-game Elo features aligned to ``games``' index.

    ``games`` must be sorted chronologically and contain home/away team, score,
    and season columns. Adds: ``home_elo_pre``, ``away_elo_pre``, ``elo_diff``
    (home advantage included), ``elo_prob`` (home win probability).
    """
    ratings: dict[str, float] = {}
    team_season: dict[str, int] = {}

    home_pre = np.empty(len(games))
    away_pre = np.empty(len(games))
    probs = np.empty(len(games))

    for i, (_, row) in enumerate(games.iterrows()):
        home, away = row["home_team"], row["away_team"]
        season = int(row["season"])

        for team in (home, away):
            r = ratings.get(team, base)
            # Revert toward the mean at each team's first game of a new season.
            if team_season.get(team) not in (None, season):
                r = base + (1 - revert) * (r - base)
            ratings[team] = r
            team_season[team] = season

        rh, ra = ratings[home], ratings[away]
        eh = _expected(rh + hfa, ra)

        home_pre[i] = rh
        away_pre[i] = ra
        probs[i] = eh

        # Update from the result.
        margin = int(row["home_score"] - row["away_score"])
        s_home = 1.0 if margin > 0 else 0.0
        if margin > 0:
            elo_diff_winner = (rh + hfa) - ra
        else:
            elo_diff_winner = ra - (rh + hfa)
        mult = _mov_multiplier(margin, elo_diff_winner)
        delta = k * mult * (s_home - eh)
        ratings[home] = rh + delta
        ratings[away] = ra - delta

    out = pd.DataFrame(index=games.index)
    out["home_elo_pre"] = home_pre
    out["away_elo_pre"] = away_pre
    out["elo_diff"] = (home_pre + hfa) - away_pre
    out["elo_prob"] = probs
    return out
