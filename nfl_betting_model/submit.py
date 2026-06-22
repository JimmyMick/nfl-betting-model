"""Write pick sheets back to the GitHub repo from the cloud app.

Streamlit Community Cloud has an ephemeral filesystem, so the in-app pick form
can't just write ``predictions/picks/*.csv`` locally — the change would vanish on
reboot and never reach the Tuesday grade cron. Instead the form commits the
sheet back to GitHub via the Contents API; the committed CSV is exactly what the
existing grader / leaderboard already read, so nothing downstream changes.

Two layers, kept separate so the merge logic is testable without a network:

* :func:`merge_player_picks` — pure: takes the current sheet text (or ``None``)
  and one player's picks, returns the new CSV text. Preserves every other
  player's rows.
* :class:`GitHubStore` — thin Contents-API client (get/put a file). Needs a
  fine-grained token with **Contents: read/write** on just this repo.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import httpx
import pandas as pd

from .picks import PICK_COLUMNS

API = "https://api.github.com"


def game_id(season: int, week: int, away: str, home: str) -> str:
    """nflverse-style id: ``2024_01_ARI_BUF`` (season_week_away_home)."""
    return f"{season}_{week:02d}_{away}_{home}"


def merge_player_picks(current_csv: str | None, season: int, week: int,
                       games: pd.DataFrame, player: str,
                       picks: dict[str, dict]) -> str:
    """Return new sheet text with ``player``'s rows for this week replaced.

    ``games`` has columns ``away_team``/``home_team`` (the week's slate, e.g. the
    exported preview). ``picks`` maps ``game_id -> {"pick": team, "confidence":
    int|""}``. Every row not belonging to ``player`` is preserved verbatim.
    """
    if current_csv and current_csv.strip():
        existing = pd.read_csv(io.StringIO(current_csv), dtype={"game_id": str})
    else:
        existing = pd.DataFrame(columns=PICK_COLUMNS)

    # Keep all rows except this player's rows for this week (we rewrite those).
    keep = existing[existing["player"].astype(str).str.strip() != player].copy()

    rows = []
    g = games.copy()
    for _, game in g.iterrows():
        gid = game_id(season, week, game["away_team"], game["home_team"])
        sel = picks.get(gid, {})
        rows.append({
            "season": season, "week": week, "game_id": gid, "gameday": "",
            "away_team": game["away_team"], "home_team": game["home_team"],
            "player": player,
            "pick": sel.get("pick", ""),
            "confidence": sel.get("confidence", ""),
        })
    mine = pd.DataFrame(rows, columns=PICK_COLUMNS)

    out = pd.concat([keep, mine], ignore_index=True)
    out = out.sort_values(["game_id", "player"]).reset_index(drop=True)
    return out.to_csv(index=False)


@dataclass
class GitHubStore:
    """Minimal GitHub Contents-API client for a single repo/branch."""

    token: str
    repo: str            # "owner/name"
    branch: str = "main"

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28"}

    def get_file(self, path: str) -> tuple[str | None, str | None]:
        """Return ``(text, sha)`` for ``path``; ``(None, None)`` if it's absent."""
        r = httpx.get(f"{API}/repos/{self.repo}/contents/{path}",
                      params={"ref": self.branch}, headers=self._headers(),
                      timeout=20)
        if r.status_code == 404:
            return None, None
        r.raise_for_status()
        data = r.json()
        text = base64.b64decode(data["content"]).decode("utf-8")
        return text, data["sha"]

    def put_file(self, path: str, text: str, message: str,
                 sha: str | None) -> None:
        """Create or update ``path`` with ``text``."""
        body = {"message": message, "branch": self.branch,
                "content": base64.b64encode(text.encode("utf-8")).decode("ascii")}
        if sha:
            body["sha"] = sha
        r = httpx.put(f"{API}/repos/{self.repo}/contents/{path}",
                      json=body, headers=self._headers(), timeout=20)
        r.raise_for_status()


def submit_picks(store: GitHubStore, season: int, week: int, games: pd.DataFrame,
                 player: str, picks: dict[str, dict]) -> None:
    """Read the week's sheet, merge in ``player``'s picks, commit it back.

    Retries once on a stale-SHA conflict (another player committing concurrently).
    """
    path = f"predictions/picks/{season}-wk{week:02d}.csv"
    for attempt in range(2):
        current, sha = store.get_file(path)
        text = merge_player_picks(current, season, week, games, player, picks)
        try:
            store.put_file(path, text, f"picks: {player} {season} wk{week:02d}", sha)
            return
        except httpx.HTTPStatusError as e:  # 409 = sha moved under us; retry once
            if e.response.status_code == 409 and attempt == 0:
                continue
            raise
