"""Load NFL teams + associated people (players, coaches, owners) into Neo4j.

Examples
--------
    uv run ingest_graph.py --seasons 2022-2023
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nflreadpy as nfl

from nfl_betting_model.graph import GraphStore

OWNERS_CSV = Path(__file__).parent / "data" / "owners.csv"


def _parse_seasons(text: str) -> list[int]:
    if "-" in text:
        lo, hi = (int(x) for x in text.split("-", 1))
        return list(range(lo, hi + 1))
    return [int(text)]


def team_rows() -> list[dict]:
    df = nfl.load_teams().to_pandas()
    return [
        {
            "abbr": r["team_abbr"],
            "name": r["team_name"],
            "nick": r["team_nick"],
            "conf": r["team_conf"],
            "division": r["team_division"],
            "team_id": r["team_id"],
        }
        for r in df.to_dict("records")
    ]


def player_and_roster_rows(seasons: list[int]) -> tuple[list[dict], list[dict]]:
    df = nfl.load_rosters(seasons=seasons).to_pandas()
    df = df[df["gsis_id"].notna()]

    players: dict[str, dict] = {}
    edges: list[dict] = []
    for r in df.to_dict("records"):
        gsis = r["gsis_id"]
        # One Player node per id (latest row wins for profile fields).
        players[gsis] = {
            "gsis_id": gsis,
            "name": r.get("full_name"),
            "position": r.get("position"),
            "position_group": r.get("ngs_position"),
            "birth_date": str(r.get("birth_date")) if r.get("birth_date") else None,
            "college": r.get("college"),
            "height": r.get("height"),
            "weight": r.get("weight"),
        }
        edges.append(
            {
                "gsis_id": gsis,
                "team": r.get("team"),
                "season": int(r["season"]),
                "position": r.get("position"),
                "jersey_number": r.get("jersey_number"),
                "status": r.get("status"),
            }
        )
    return list(players.values()), edges


def coach_rows(seasons: list[int]) -> list[dict]:
    df = nfl.load_schedules(seasons=seasons).to_pandas()
    seen: set[tuple] = set()
    rows: list[dict] = []
    for r in df.to_dict("records"):
        for side in ("home", "away"):
            name = r.get(f"{side}_coach")
            team = r.get(f"{side}_team")
            season = r.get("season")
            if not name or not team:
                continue
            key = (name, team, season)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {"name": name, "team": team, "season": int(season), "role": "HC"}
            )
    return rows


def owner_rows() -> list[dict]:
    """Load ownership from the user-maintained CSV, if present."""
    if not OWNERS_CSV.exists():
        return []
    with OWNERS_CSV.open() as fh:
        return [
            {"name": r["name"], "team": r["team"], "role": r.get("role") or "Owner"}
            for r in csv.DictReader(fh)
            if r.get("name") and r.get("team")
        ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest NFL graph into Neo4j")
    ap.add_argument("--seasons", default="2023", help="e.g. 2022-2023 or 2023")
    args = ap.parse_args()
    seasons = _parse_seasons(args.seasons)

    with GraphStore() as g:
        print(f"Connecting to {g.uri} ...")
        g.verify()
        g.setup_constraints()

        teams = team_rows()
        print(f"Teams: {len(teams)}")
        g.ingest_teams(teams)

        players, edges = player_and_roster_rows(seasons)
        print(f"Players: {len(players)}  roster edges: {len(edges)}")
        g.ingest_players(players)
        g.ingest_roster_edges(edges)

        coaches = coach_rows(seasons)
        print(f"Coach-season edges: {len(coaches)}")
        g.ingest_coaches(coaches)

        owners = owner_rows()
        print(f"Owners (from CSV): {len(owners)}")
        g.ingest_owners(owners)

        print("\nGraph counts:")
        for label, n in g.counts().items():
            print(f"  {label:12s} {n}")


if __name__ == "__main__":
    main()
