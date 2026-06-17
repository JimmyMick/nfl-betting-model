"""Game-time weather (wind, temperature, indoor flag) per game.

Unlike every other feature block, weather is a *game-level* attribute: the same
wind and temperature apply to both teams, so there is no home-minus-away diff to
take. Any win-probability signal therefore has to come from an *interaction* —
wind suppresses the passing game, which compresses the stronger (more
pass-dependent favorite's) edge toward a coin flip. ``features.build_features``
builds that interaction (``wind_x_edge``); this module just supplies the raw
per-game conditions.

Source: nflreadpy ``load_schedules`` carries ``roof``, ``temp`` and ``wind``.
Indoor games (dome/closed roof) have no weather — that is a real signal
(controlled conditions), so they are set to neutral (wind 0, mild temp). Open
retractable roofs and outdoors are treated as exposed to weather.

Leakage caveat: ``temp``/``wind`` here are the *actual recorded* game-time
values. For historical training that is the best available proxy for what a
forecast would have shown; a bettor a few days out holds a forecast, not the
exact reading, but temp/wind forecasts are accurate enough that the train/serve
gap is small. The live preview path (``predict.py``) would substitute a forecast
— wired only if this feature proves out in walk-forward validation.

NULL / THIN RESULT (sigmoid walk-forward 2021-2025, both Elo+EPA and the full
+QB+Starters set; see ``validate_weather.py`` and ``validate_climate.py``):

  Raw weather (``wind``, ``cold``, ``wind_x_edge``) — a *thin, likely-priced*
  signal: 4/5 seasons improve logloss and brier but by tiny margins (e.g.
  0.6308→0.6288), an order of magnitude smaller than the availability feature,
  and AUC is only 3/5. Not worth building live forecast plumbing for.

  Climate mismatch (``climate_mismatch_diff``, the "warm/dome team yanked into
  the cold" acclimation spot) — a clean **null**: only 2/5 seasons improve,
  uniformly across logloss/brier/AUC and every feature tier (coin-flip-or-worse).
  The more precisely it targets the dome-team-in-the-cold trope, the more it
  washes out — because that is the oldest, most thoroughly priced angle in the
  market. The feature is correct (top mismatches are MIN@GB, DAL@BUF, LV@CHI);
  the line simply already ate it.

Verdict: weather is the 4th documented dead-end (after PFF, coaching, QB-EPA) —
external/conditions proxies keep washing out because the moneyline market prices
everything knowable in advance. Only *availability* (who is literally not suited
up) has ever been orthogonal enough to add signal. Kept here, gated and out of
the live preview path, so it isn't re-tried from scratch. Run via ``main.py``
(default on; ``--no-weather`` to skip).
"""

from __future__ import annotations

import nflreadpy as nfl
import pandas as pd

# Roof values that seal the field off from weather.
INDOOR_ROOFS = {"dome", "closed"}
# Neutral conditions assigned to indoor games (no wind, mild temperature).
INDOOR_WIND = 0.0
INDOOR_TEMP = 70.0
# Fallback imputation for outdoor games whose weather wasn't recorded (a known
# nflverse backfill gap in 2022-2023). League-typical conditions.
DEFAULT_OUTDOOR_TEMP = 60.0
DEFAULT_OUTDOOR_WIND = 8.0


def game_weather(seasons: list[int]) -> pd.DataFrame:
    """Return ``[game_id, wind, temp, indoor]`` — one row per scheduled game.

    Indoor games get neutral conditions; outdoor games missing a recorded value
    are imputed to the outdoor median (or a league-typical constant when no
    outdoor data is available), so callers never see NaN.
    """
    raw = nfl.load_schedules(seasons=list(seasons))
    df = raw.to_pandas() if hasattr(raw, "to_pandas") else pd.DataFrame(raw)

    cols = [c for c in ("game_id", "roof", "temp", "wind") if c in df.columns]
    df = df[cols].copy()
    for c in ("temp", "wind"):
        if c not in df.columns:
            df[c] = pd.NA
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["indoor"] = df.get("roof", pd.Series(index=df.index)).isin(INDOOR_ROOFS)

    outdoor = ~df["indoor"]
    # Impute missing outdoor readings from the populated outdoor median.
    temp_fill = df.loc[outdoor, "temp"].median()
    wind_fill = df.loc[outdoor, "wind"].median()
    temp_fill = DEFAULT_OUTDOOR_TEMP if pd.isna(temp_fill) else temp_fill
    wind_fill = DEFAULT_OUTDOOR_WIND if pd.isna(wind_fill) else wind_fill

    df.loc[outdoor, "temp"] = df.loc[outdoor, "temp"].fillna(temp_fill)
    df.loc[outdoor, "wind"] = df.loc[outdoor, "wind"].fillna(wind_fill)

    # Indoor games: override to neutral regardless of any stray recorded value.
    df.loc[df["indoor"], "temp"] = INDOOR_TEMP
    df.loc[df["indoor"], "wind"] = INDOOR_WIND

    return df[["game_id", "wind", "temp", "indoor"]].reset_index(drop=True)


# --- Climate mismatch (acclimation) ------------------------------------------
# Degrees of "temp drop" that count as one mismatch unit, and mph of "excess
# wind" per unit. Scaled so a dome team in a 20F / 15mph game lands around 5-6.
TEMP_DROP_PER_UNIT = 10.0
WIND_EXCESS_PER_UNIT = 5.0


def climate_mismatch(seasons: list[int]) -> pd.DataFrame:
    """Return ``[game_id, climate_mismatch_diff]`` — the acclimation spot.

    Each team has a *home climate baseline* — the median temp/wind across its own
    home games (dome teams get controlled 70F / 0mph). A team pays a mismatch
    penalty when the game it plays in is colder and/or windier than what it is
    built for; the penalty grows with how far conditions fall outside its norm:

        penalty(team) = max(0, home_temp - game_temp)/10
                      + max(0, game_wind - home_wind)/5

    ``climate_mismatch_diff = away_penalty - home_penalty`` is oriented so a
    positive value favors the home team (the away side is the one yanked out of
    its element; the home side is, by definition, at its baseline so its penalty
    is ~0). This is the *differential acclimation* signal — distinct from raw
    cold/wind, which apply equally to both teams. Geography is a fixed stadium
    constant, so using full-sample home baselines is not performance leakage.
    """
    cond = game_weather(seasons)  # [game_id, wind, temp, indoor]

    raw = nfl.load_schedules(seasons=list(seasons))
    sched = raw.to_pandas() if hasattr(raw, "to_pandas") else pd.DataFrame(raw)
    sched = sched[["game_id", "home_team", "away_team"]].merge(cond, on="game_id")

    # Per-team home baseline from its own home games' actual conditions.
    home_rows = sched[["home_team", "temp", "wind", "indoor"]].rename(
        columns={"home_team": "team"}
    )
    base = home_rows.groupby("team").agg(
        home_temp=("temp", "median"), home_wind=("wind", "median")
    )

    def _penalty(team_col: str) -> "pd.Series":
        b = base.reindex(sched[team_col].to_numpy())
        home_temp = b["home_temp"].to_numpy()
        home_wind = b["home_wind"].to_numpy()
        temp_drop = (home_temp - sched["temp"].to_numpy()).clip(min=0)
        wind_excess = (sched["wind"].to_numpy() - home_wind).clip(min=0)
        return temp_drop / TEMP_DROP_PER_UNIT + wind_excess / WIND_EXCESS_PER_UNIT

    sched["climate_mismatch_diff"] = _penalty("away_team") - _penalty("home_team")
    return sched[["game_id", "climate_mismatch_diff"]].reset_index(drop=True)
