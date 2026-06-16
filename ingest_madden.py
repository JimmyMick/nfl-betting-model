"""Attach Madden ratings to PLAYED_FOR edges in the Neo4j graph.

Run after ``ingest_graph.py`` (players/edges must exist first).

Examples
--------
    uv run ingest_madden.py --seasons 2023
    uv run ingest_madden.py --seasons 2020-2023
"""

from __future__ import annotations

import argparse

from nfl_betting_model import madden
from nfl_betting_model.graph import GraphStore


def _parse_seasons(text: str) -> list[int]:
    if "-" in text:
        lo, hi = (int(x) for x in text.split("-", 1))
        return list(range(lo, hi + 1))
    return [int(text)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest Madden ratings into Neo4j")
    ap.add_argument("--seasons", default="2023", help="e.g. 2020-2023 or 2023")
    args = ap.parse_args()
    seasons = _parse_seasons(args.seasons)

    ratings = madden.load_ratings(seasons)
    print(f"Loaded {len(ratings)} Madden ratings across seasons {seasons}")

    with GraphStore() as g:
        print(f"Connecting to {g.uri} ...")
        g.verify()
        rows = ratings.to_dict("records")
        n_edges = g.ingest_madden_ratings(rows)
        print(f"Set madden_ovr on {n_edges} PLAYED_FOR edges")


if __name__ == "__main__":
    main()
