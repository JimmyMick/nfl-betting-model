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
SIM_FILE = "playoff_odds.csv"
SIM_HISTORY_FILE = "playoff_odds_history.csv"
BLOG_DIR = "blog"
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


SIM_COLS = [
    "team", "conference", "division", "wins_now", "games_played", "proj_wins",
    "make_playoffs", "win_division", "top_seed", "win_conference", "win_sb",
]


def write_sim_artifacts(proj: pd.DataFrame, season: int, through_week: int,
                        out_dir: Path = ARTIFACT_DIR) -> Path:
    """Export the latest playoff-odds projection plus an appended weekly history.

    The current snapshot (``playoff_odds.csv``) is overwritten each run. A
    ``through_week``-stamped copy is appended to ``playoff_odds_history.csv`` so a
    team's odds can be tracked week to week; re-running the same week replaces
    that week's rows (idempotent).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [c for c in SIM_COLS if c in proj.columns]
    snap = proj[cols].copy()
    snap.to_csv(out_dir / SIM_FILE, index=False)

    stamped = snap.copy()
    stamped.insert(0, "through_week", int(through_week))
    stamped.insert(0, "season", int(season))
    hist_path = out_dir / SIM_HISTORY_FILE
    if hist_path.exists():
        prev = pd.read_csv(hist_path)
        prev = prev[~((prev["season"] == season)
                      & (prev["through_week"] == through_week))]
        stamped = pd.concat([prev, stamped], ignore_index=True)
    stamped.to_csv(hist_path, index=False)

    _write_meta(out_dir, sim_season=int(season), sim_through_week=int(through_week),
                sim_generated_at=_now())
    return out_dir


def _slugify(text: str) -> str:
    import re
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (slug[:60] or "post")


def save_blog_post(title: str, author: str, body: str,
                   out_dir: Path = ARTIFACT_DIR,
                   when: dt.datetime | None = None) -> Path:
    """Write a blog post as ``blog/YYYYMMDD-HHMMSS-slug.md`` with YAML frontmatter.

    Reverse-lex filenames sort as reverse-chronological on disk, so the reader
    doesn't need to parse frontmatter dates just to order the feed.
    """
    when = when or dt.datetime.now()
    blog_dir = out_dir / BLOG_DIR
    blog_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{when.strftime('%Y%m%d-%H%M%S')}-{_slugify(title)}.md"
    path = blog_dir / fname
    body = (body or "").rstrip() + "\n"
    frontmatter = (
        "---\n"
        f"title: {title}\n"
        f"author: {author}\n"
        f"date: {when.strftime('%Y-%m-%d %H:%M')}\n"
        "---\n\n"
    )
    path.write_text(frontmatter + body)
    return path


def _parse_post(path: Path) -> dict:
    text = path.read_text()
    meta = {"title": path.stem, "author": "", "date": ""}
    body = text
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            for line in text[4:end].splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip().lower()] = v.strip()
            body = text[end + 4:].lstrip("\n")
    meta["body"] = body
    meta["path"] = str(path)
    meta["filename"] = path.name
    return meta


def load_blog_posts(art_dir: Path = ARTIFACT_DIR) -> list[dict]:
    """Return all blog posts, most recent first (by filename)."""
    blog_dir = art_dir / BLOG_DIR
    if not blog_dir.exists():
        return []
    return [_parse_post(p) for p in sorted(blog_dir.glob("*.md"), reverse=True)]


def delete_blog_post(filename: str, art_dir: Path = ARTIFACT_DIR) -> bool:
    path = art_dir / BLOG_DIR / filename
    if path.exists() and path.parent == art_dir / BLOG_DIR:
        path.unlink()
        return True
    return False


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
        "sim": _maybe(SIM_FILE),
        "sim_history": _maybe(SIM_HISTORY_FILE),
        "blog": load_blog_posts(art_dir),
        "meta": _read_meta(art_dir),
    }
