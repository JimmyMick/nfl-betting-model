"""Expert pick tracking: collect human picks each week and score them next to
the model and the market on identical games.

The companion to the model's own scorekeeping (grade.py). Each participant
submits, per game, a **winner** and a **confidence** (50-100). Picks live in
``predictions/picks/{season}-wk{week:02d}.csv`` — one row per (game, player),
seeded blank from that week's schedule and filled in before kickoff. The grader
joins them to completed games and scores everyone on the same footing the model
gets: straight-up accuracy plus calibration (Brier / log loss) whenever a
confidence is supplied.

This module is deliberately network-free and pandas-only so it's easy to test;
the schedule fetch for seeding lives in the picks.py CLI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

# predictions/picks/ at the repo root (this file is repo/nfl_betting_model/picks.py).
PICKS_DIR = Path(__file__).resolve().parent.parent / "predictions" / "picks"
PLAYERS_FILE = PICKS_DIR / "players.txt"

# Columns in a weekly pick sheet. game_id is the join key; pick/confidence are
# the only two a human fills in. ``rationale`` is optional free text (the LLM
# "AI expert" writes a one-liner per pick; humans leave it blank) — metadata
# only, never used in scoring.
PICK_COLUMNS = [
    "season", "week", "game_id", "gameday",
    "away_team", "home_team", "player", "pick", "confidence", "rationale",
]


def week_path(season: int, week: int) -> Path:
    """Path to the weekly pick sheet for ``season``/``week``."""
    return PICKS_DIR / f"{season}-wk{week:02d}.csv"


def load_players(path: Path = PLAYERS_FILE) -> list[str]:
    """Participant names, one per line. Blank lines and ``#`` comments ignored."""
    if not path.exists():
        return []
    names = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            names.append(line)
    return names


def seed_week(games_week: pd.DataFrame, players: list[str], out_path: Path,
              force: bool = False) -> tuple[Path, bool]:
    """Write a blank pick sheet: one row per (game, player), pick/confidence empty.

    Returns ``(path, written)``. Refuses to overwrite an existing sheet unless
    ``force`` — a sheet may already hold filled picks, and clobbering it would
    erase the record.
    """
    if out_path.exists() and not force:
        return out_path, False
    if not players:
        raise ValueError(
            "No players to seed. Add names to predictions/picks/players.txt."
        )

    g = games_week.sort_values(["gameday", "game_id"])
    rows = []
    for _, game in g.iterrows():
        gameday = pd.to_datetime(game["gameday"], errors="coerce")
        for player in players:
            rows.append({
                "season": int(game["season"]),
                "week": int(game["week"]),
                "game_id": game["game_id"],
                "gameday": gameday.date().isoformat() if pd.notna(gameday) else "",
                "away_team": game["away_team"],
                "home_team": game["home_team"],
                "player": player,
                "pick": "",
                "confidence": "",
                "rationale": "",
            })
    out = pd.DataFrame(rows, columns=PICK_COLUMNS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out_path, True


def load_week_picks(path: Path) -> pd.DataFrame:
    """Read one filled sheet, keeping only rows with a non-blank pick."""
    df = pd.read_csv(path, dtype={"game_id": str})
    if "rationale" not in df.columns:  # tolerate pre-rationale sheets
        df["rationale"] = ""
    df["pick"] = df["pick"].astype("string").str.strip()
    df = df[df["pick"].notna() & (df["pick"] != "")].copy()
    df["player"] = df["player"].astype("string").str.strip()
    return df


def load_all_picks(season: int, through_week: int,
                   picks_dir: Path = PICKS_DIR) -> pd.DataFrame:
    """Every filled pick for ``season`` from week 1 through ``through_week``."""
    frames = []
    for wk in range(1, through_week + 1):
        path = picks_dir / f"{season}-wk{wk:02d}.csv"
        if path.exists():
            wk_df = load_week_picks(path)
            if not wk_df.empty:
                frames.append(wk_df)
    if not frames:
        return pd.DataFrame(columns=PICK_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def score(picks: pd.DataFrame, graded: pd.DataFrame) -> pd.DataFrame:
    """Join picks to graded games and score each one.

    ``graded`` is grade.grade_season() output (one row per played game with
    ``winner``, ``home_win``, ``model_correct``). Returns one row per scored pick
    with ``correct`` (straight-up) and ``player_home_prob`` (the confidence
    re-expressed as a home-win probability, NaN when no confidence was given).
    Picks whose team isn't in the matchup are dropped with a warning.
    """
    g = graded[["game_id", "week", "home_team", "away_team",
                "home_win", "winner", "model_correct"]]
    m = picks.merge(g, on="game_id", how="inner", suffixes=("", "_g"))

    pick = m["pick"].astype("string").str.upper().str.strip()
    home = m["home_team_g"].astype("string").str.upper()
    away = m["away_team_g"].astype("string").str.upper()
    valid = pick.eq(home) | pick.eq(away)
    if (~valid).any():
        bad = m.loc[~valid, ["player", "game_id", "pick"]]
        for _, r in bad.iterrows():
            print(f"  ! dropping {r['player']}'s pick '{r['pick']}' for "
                  f"{r['game_id']} — not a team in that game.")
    m = m[valid].copy()
    if m.empty:
        return m

    pick = pick[valid]
    home = home[valid]
    winner = m["winner"].astype("string").str.upper()
    m["pick"] = pick.to_numpy()
    m["correct"] = (pick.to_numpy() == winner.to_numpy()).astype(int)

    conf = pd.to_numeric(m["confidence"], errors="coerce")
    p_team = conf / 100.0  # implied probability that the *picked* team wins
    m["player_home_prob"] = np.where(pick.to_numpy() == home.to_numpy(),
                                     p_team, 1 - p_team)
    return m


def _record(correct: pd.Series) -> str:
    """e.g. '10-6 (63%)' from a 0/1 correctness series."""
    w = int(correct.sum())
    n = len(correct)
    return f"{w}-{n - w} ({w / n:.0%})" if n else "0-0 (—)"


def leaderboard(scored: pd.DataFrame, graded: pd.DataFrame) -> pd.DataFrame:
    """Season standings: each player vs the model **on the games they picked**.

    Columns: Player, Picks, Record, vs Model (player accuracy minus the model's
    accuracy over that same subset), Brier, Log loss. Calibration columns use
    only picks that carried a confidence.
    """
    rows = []
    for player, grp in scored.groupby("player"):
        n = len(grp)
        sub = graded[graded["game_id"].isin(grp["game_id"])]
        model_acc = sub["model_correct"].mean() if len(sub) else float("nan")
        delta = grp["correct"].mean() - model_acc

        cal = grp.dropna(subset=["player_home_prob"])
        if len(cal):
            br = f"{brier_score_loss(cal['home_win'], cal['player_home_prob']):.3f}"
            ll = f"{log_loss(cal['home_win'], cal['player_home_prob'], labels=[0, 1]):.3f}"
        else:
            br = ll = "—"

        rows.append({
            "Player": player,
            "Picks": n,
            "Record": _record(grp["correct"]),
            "vs Model": f"{delta:+.0%}" if pd.notna(delta) else "—",
            "Brier": br,
            "Log loss": ll,
            "_acc": grp["correct"].mean(),
        })
    out = pd.DataFrame(rows).sort_values("_acc", ascending=False)
    return out.drop(columns="_acc").reset_index(drop=True)
