"""Build strictly pre-game features (no leakage from the game being predicted).

The feature set is composable: base recent-form features are always built, with
optional Elo and EPA blocks layered on. ``build_features`` returns the frame and
the list of active feature columns so callers can compare feature sets.
"""

from __future__ import annotations

import pandas as pd

from . import data as data_mod

# Rolling window (number of prior games) for recent-form features.
FORM_WINDOW = 5

BASE_FEATURES = [
    "form_pf_diff",       # home recent points-for minus away
    "form_pa_diff",       # home recent points-against minus away
    "form_margin_diff",   # home recent point margin minus away
    "form_winrate_diff",  # home recent win rate minus away
    "season_winrate_diff",
    "rest_diff",
    "div_game",
]
ELO_FEATURES = ["elo_diff", "elo_prob"]
EPA_FEATURES = ["off_epa_diff", "def_epa_diff", "net_epa_diff"]
MADDEN_FEATURES = ["qb_ovr_diff"]
QB_EPA_FEATURES = ["qb_epa_diff"]
STARTER_FEATURES = ["ol_ovr_diff", "dl_ovr_diff", "db_ovr_diff", "starter_ovr_diff"]
COACH_FEATURES = ["coach_winpct_diff", "coach_new_diff"]


def _roll(frame: pd.DataFrame, col: str) -> pd.Series:
    """Per-team rolling mean of ``col`` using only prior games (shift(1))."""
    return frame.groupby("team", group_keys=False)[col].apply(
        lambda s: s.shift(1).rolling(FORM_WINDOW, min_periods=1).mean()
    )


def _add_team_form(long: pd.DataFrame, extra_cols: list[str]) -> pd.DataFrame:
    long["form_pf"] = _roll(long, "points_for")
    long["form_pa"] = _roll(long, "points_against")
    long["form_margin"] = long["form_pf"] - long["form_pa"]
    long["form_winrate"] = _roll(long, "won")
    long["season_winrate"] = long.groupby(["team", "season"], group_keys=False)[
        "won"
    ].apply(lambda s: s.shift(1).expanding().mean())
    for col in extra_cols:
        long[f"{col}_form"] = _roll(long, col)
    return long


def build_features(
    games: pd.DataFrame,
    epa_table: pd.DataFrame | None = None,
    elo_table: pd.DataFrame | None = None,
    qb_table: pd.DataFrame | None = None,
    qb_epa_table: pd.DataFrame | None = None,
    starter_table: pd.DataFrame | None = None,
    coach_table: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Return ``(features_df, feature_cols)`` for the given games."""
    long = data_mod.to_long(games)

    extra = []
    if epa_table is not None:
        long = long.merge(epa_table, on=["game_id", "team"], how="left")
        extra = ["off_epa", "def_epa"]

    long = _add_team_form(long, extra)

    form_cols = ["form_pf", "form_pa", "form_margin", "form_winrate", "season_winrate"]
    form_cols += [f"{c}_form" for c in extra]
    keyed = long.set_index(["game_id", "team"])[form_cols]

    df = games.copy()

    def _lookup(team_col: str, suffix: str) -> None:
        idx = pd.MultiIndex.from_arrays([df["game_id"], df[team_col]])
        for c in form_cols:
            df[f"{c}_{suffix}"] = keyed[c].reindex(idx).to_numpy()

    _lookup("home_team", "home")
    _lookup("away_team", "away")

    # Base form diffs.
    df["form_pf_diff"] = df["form_pf_home"] - df["form_pf_away"]
    df["form_pa_diff"] = df["form_pa_home"] - df["form_pa_away"]
    df["form_margin_diff"] = df["form_margin_home"] - df["form_margin_away"]
    df["form_winrate_diff"] = df["form_winrate_home"] - df["form_winrate_away"]
    df["season_winrate_diff"] = df["season_winrate_home"] - df["season_winrate_away"]
    df["rest_diff"] = df["home_rest"].fillna(7) - df["away_rest"].fillna(7)
    df["div_game"] = df["div_game"].fillna(0).astype(float)

    feature_cols = list(BASE_FEATURES)

    if epa_table is not None:
        df["off_epa_diff"] = df["off_epa_form_home"] - df["off_epa_form_away"]
        df["def_epa_diff"] = df["def_epa_form_home"] - df["def_epa_form_away"]
        # Net strength = offense minus defense allowed, home edge over away.
        home_net = df["off_epa_form_home"] - df["def_epa_form_home"]
        away_net = df["off_epa_form_away"] - df["def_epa_form_away"]
        df["net_epa_diff"] = home_net - away_net
        feature_cols += EPA_FEATURES

    if elo_table is not None:
        df = df.join(elo_table[ELO_FEATURES])
        feature_cols += ELO_FEATURES

    if qb_table is not None:
        # Starting-QB Madden OVR per (game, team) -> home minus away.
        keyed_qb = qb_table.set_index(["game_id", "team"])["qb_ovr"]
        home_idx = pd.MultiIndex.from_arrays([df["game_id"], df["home_team"]])
        away_idx = pd.MultiIndex.from_arrays([df["game_id"], df["away_team"]])
        df["qb_ovr_home"] = keyed_qb.reindex(home_idx).to_numpy()
        df["qb_ovr_away"] = keyed_qb.reindex(away_idx).to_numpy()
        df["qb_ovr_diff"] = df["qb_ovr_home"] - df["qb_ovr_away"]
        feature_cols += MADDEN_FEATURES

    if qb_epa_table is not None:
        # Starting QB's rolling prior passing EPA/play -> home minus away.
        keyed_qe = qb_epa_table.set_index(["game_id", "team"])["qb_epa"]
        home_idx = pd.MultiIndex.from_arrays([df["game_id"], df["home_team"]])
        away_idx = pd.MultiIndex.from_arrays([df["game_id"], df["away_team"]])
        df["qb_epa_home"] = keyed_qe.reindex(home_idx).to_numpy()
        df["qb_epa_away"] = keyed_qe.reindex(away_idx).to_numpy()
        df["qb_epa_diff"] = df["qb_epa_home"] - df["qb_epa_away"]
        feature_cols += QB_EPA_FEATURES

    if starter_table is not None:
        # Starting-unit Madden OVR (OL/DL/secondary/overall) -> home minus away.
        st = starter_table.set_index(["game_id", "team"])
        home_idx = pd.MultiIndex.from_arrays([df["game_id"], df["home_team"]])
        away_idx = pd.MultiIndex.from_arrays([df["game_id"], df["away_team"]])
        for unit in ("ol_ovr", "dl_ovr", "db_ovr", "starter_ovr"):
            home = st[unit].reindex(home_idx).to_numpy()
            away = st[unit].reindex(away_idx).to_numpy()
            df[f"{unit}_diff"] = home - away
        feature_cols += STARTER_FEATURES

    if coach_table is not None:
        # Coach career win% + new-regime flag -> home minus away.
        ct = coach_table.set_index(["game_id", "team"])
        home_idx = pd.MultiIndex.from_arrays([df["game_id"], df["home_team"]])
        away_idx = pd.MultiIndex.from_arrays([df["game_id"], df["away_team"]])
        for col in ("coach_winpct", "coach_new"):
            home = ct[col].reindex(home_idx).to_numpy()
            away = ct[col].reindex(away_idx).to_numpy()
            df[f"{col}_diff"] = home - away
        feature_cols += COACH_FEATURES

    # Drop rows with no prior-form signal on either side.
    df = df.dropna(subset=["form_margin_diff", "form_winrate_diff"]).reset_index(
        drop=True
    )
    return df, feature_cols


# --- Market benchmark helpers -------------------------------------------------

def american_to_prob(odds: pd.Series) -> pd.Series:
    """Convert American moneyline odds to implied probability."""
    import numpy as np

    odds = pd.to_numeric(odds, errors="coerce")
    return np.where(odds < 0, -odds / (-odds + 100.0), 100.0 / (odds + 100.0))


def market_home_prob(games: pd.DataFrame) -> pd.Series:
    """Vig-free implied home win probability from the moneyline pair."""
    ph = american_to_prob(games["home_moneyline"])
    pa = american_to_prob(games["away_moneyline"])
    total = ph + pa
    return pd.Series(ph / total, index=games.index)
