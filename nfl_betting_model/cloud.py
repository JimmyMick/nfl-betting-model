"""Lightweight artifact layer for the cloud (read-only) dashboard.

The full pipeline trains on ~15 seasons of play-by-play, which is too heavy for
Streamlit Community Cloud's ~1 GB free tier. Instead the local weekly runs
(predict.py / grade.py, already training) *export* their results here as small
CSVs, commit + push them, and the cloud app (`streamlit_app.py`) just renders
these — no training, no nflreadpy fetch, no Madden data needed in the cloud.

This module is deliberately dependency-light (pandas + stdlib) so the cloud
requirements stay tiny.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

# predictions/cloud/ at the repo root (this file is repo/nfl_betting_model/cloud.py).
ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "predictions" / "cloud"

GRADED_FILE = "graded_games.csv"
SCORED_FILE = "scored_picks.csv"
PREVIEW_FILE = "latest_preview.csv"
SCHEDULE_FILE = "schedule.csv"
META_FILE = "meta.json"

# Columns each artifact carries — kept explicit so the cloud reader and the
# exporters can't drift apart.
GRADED_COLS = [
    "game_id", "week", "home_team", "away_team", "model_home_prob",
    "market_home_prob", "home_win", "winner", "model_pick", "model_correct",
    "market_correct",
]
SCORED_COLS = [
    "player", "game_id", "week", "home_team", "away_team", "pick", "correct",
    "player_home_prob", "home_win", "winner", "model_correct", "rationale",
]
PREVIEW_COLS = [
    "home_team", "away_team", "model_home_prob", "market_home_prob", "edge",
    "driver", "home_win",
]
# Full-season matchup schedule (dependency-light; no model output). Scores stay
# blank for unplayed games and fill in as the weekly runs refresh the artifact.
SCHEDULE_COLS = [
    "week", "game_type", "gameday", "away_team", "home_team",
    "away_score", "home_score",
]


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _read_meta(out_dir: Path) -> dict:
    path = out_dir / META_FILE
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _write_meta(out_dir: Path, **updates) -> None:
    meta = _read_meta(out_dir)
    meta.update(updates)
    (out_dir / META_FILE).write_text(json.dumps(meta, indent=2) + "\n")


def write_grade_artifacts(graded: pd.DataFrame, scored: pd.DataFrame | None,
                          season: int, through_week: int,
                          out_dir: Path = ARTIFACT_DIR) -> Path:
    """Export the season grade (and any scored picks) for the cloud dashboard."""
    out_dir.mkdir(parents=True, exist_ok=True)
    graded[[c for c in GRADED_COLS if c in graded.columns]].to_csv(
        out_dir / GRADED_FILE, index=False)

    has_picks = scored is not None and not scored.empty
    cols = [c for c in SCORED_COLS if scored is not None and c in scored.columns]
    frame = scored[cols] if has_picks else pd.DataFrame(columns=SCORED_COLS)
    frame.to_csv(out_dir / SCORED_FILE, index=False)

    _write_meta(out_dir, grade_season=int(season),
                grade_through_week=int(through_week),
                grade_generated_at=_now(), has_picks=bool(has_picks))
    return out_dir


def write_preview_artifacts(target: pd.DataFrame, season: int, week: int,
                            out_dir: Path = ARTIFACT_DIR) -> Path:
    """Export the latest weekly preview slate for the cloud dashboard."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target[[c for c in PREVIEW_COLS if c in target.columns]].to_csv(
        out_dir / PREVIEW_FILE, index=False)
    _write_meta(out_dir, preview_season=int(season), preview_week=int(week),
                preview_generated_at=_now())
    return out_dir


def write_schedule_artifacts(games: pd.DataFrame, season: int,
                             out_dir: Path = ARTIFACT_DIR) -> Path:
    """Export the full-season matchup schedule for the cloud dashboard.

    ``games`` is any frame with the standard schedule columns (e.g. the frame
    from ``data.load_games(..., include_unplayed=True)``); regular-season and
    playoff games are kept, preseason dropped. Final scores are written when a
    game has been played and left blank otherwise, so re-exporting each week
    keeps results current without any model output.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sched = games[games["season"] == season].copy()
    if "game_type" in sched.columns:
        sched = sched[sched["game_type"].isin(["REG", "POST"])]
    if "gameday" in sched.columns:
        sched["gameday"] = pd.to_datetime(
            sched["gameday"], errors="coerce").dt.strftime("%Y-%m-%d")
    cols = [c for c in SCHEDULE_COLS if c in sched.columns]
    sched = sched[cols].sort_values(
        [c for c in ("week", "gameday") if c in cols])
    sched.to_csv(out_dir / SCHEDULE_FILE, index=False)
    _write_meta(out_dir, schedule_season=int(season),
                schedule_generated_at=_now())
    return out_dir


def load_artifacts(art_dir: Path = ARTIFACT_DIR) -> dict:
    """Read whatever artifacts exist. Missing frames come back as ``None``."""
    def _maybe(name: str) -> pd.DataFrame | None:
        path = art_dir / name
        if not path.exists():
            return None
        df = pd.read_csv(path, dtype={"game_id": str})
        return df if not df.empty else None

    return {
        "graded": _maybe(GRADED_FILE),
        "scored": _maybe(SCORED_FILE),
        "preview": _maybe(PREVIEW_FILE),
        "schedule": _maybe(SCHEDULE_FILE),
        "meta": _read_meta(art_dir),
    }
