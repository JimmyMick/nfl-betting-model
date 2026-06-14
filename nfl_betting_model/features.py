"""Build strictly pre-game features (no leakage from the game being predicted)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as data_mod

# Rolling window (number of prior games) for recent-form features.
FORM_WINDOW = 5

FEATURE_COLS = [
    "form_pf_diff",      # home recent points-for minus away recent points-for
    "form_pa_diff",      # home recent points-against minus away recent points-against
    "form_margin_diff",  # home recent point margin minus away recent point margin
    "form_winrate_diff", # home recent win rate minus away recent win rate
    "season_winrate_diff",  # season-to-date win rate, home minus away
    "rest_diff",         # home_rest minus away_rest
    "div_game",          # divisional matchup flag
]


def _add_team_form(long: pd.DataFrame) -> pd.DataFrame:
    """Add rolling/expanding form columns to the long team-game frame.

    All stats are ``shift(1)`` first so a game never sees its own result —
    every feature uses only games that finished before kickoff.
    """
    g = long.groupby("team", group_keys=False)

    def _roll(col: str) -> pd.Series:
        return g[col].apply(
            lambda s: s.shift(1).rolling(FORM_WINDOW, min_periods=1).mean()
        )

    long["form_pf"] = _roll("points_for")
    long["form_pa"] = _roll("points_against")
    long["form_margin"] = long["form_pf"] - long["form_pa"]
    long["form_winrate"] = _roll("won")

    # Season-to-date win rate (expanding within season, shifted).
    long["season_winrate"] = long.groupby(["team", "season"], group_keys=False)[
        "won"
    ].apply(lambda s: s.shift(1).expanding().mean())

    return long


def build_features(games: pd.DataFrame) -> pd.DataFrame:
    """Return ``games`` augmented with model-ready feature columns + target."""
    long = data_mod.to_long(games)
    long = _add_team_form(long)

    form_cols = ["form_pf", "form_pa", "form_margin", "form_winrate", "season_winrate"]
    keyed = long.set_index(["game_id", "team"])[form_cols]

    df = games.copy()

    def _lookup(team_col: str, suffix: str) -> None:
        idx = pd.MultiIndex.from_arrays([df["game_id"], df[team_col]])
        for c in form_cols:
            df[f"{c}_{suffix}"] = keyed[c].reindex(idx).to_numpy()

    _lookup("home_team", "home")
    _lookup("away_team", "away")

    df["form_pf_diff"] = df["form_pf_home"] - df["form_pf_away"]
    df["form_pa_diff"] = df["form_pa_home"] - df["form_pa_away"]
    df["form_margin_diff"] = df["form_margin_home"] - df["form_margin_away"]
    df["form_winrate_diff"] = df["form_winrate_home"] - df["form_winrate_away"]
    df["season_winrate_diff"] = df["season_winrate_home"] - df["season_winrate_away"]

    df["rest_diff"] = df["home_rest"].fillna(7) - df["away_rest"].fillna(7)
    df["div_game"] = df["div_game"].fillna(0).astype(float)

    # Drop early-season rows with no prior-form signal on either side.
    df = df.dropna(subset=["form_margin_diff", "form_winrate_diff"]).reset_index(
        drop=True
    )
    return df


# --- Market benchmark helpers -------------------------------------------------

def american_to_prob(odds: pd.Series) -> pd.Series:
    """Convert American moneyline odds to implied probability."""
    odds = pd.to_numeric(odds, errors="coerce")
    return np.where(odds < 0, -odds / (-odds + 100.0), 100.0 / (odds + 100.0))


def market_home_prob(games: pd.DataFrame) -> pd.Series:
    """Vig-free implied home win probability from the moneyline pair."""
    ph = american_to_prob(games["home_moneyline"])
    pa = american_to_prob(games["away_moneyline"])
    total = ph + pa
    return pd.Series(ph / total, index=games.index)
