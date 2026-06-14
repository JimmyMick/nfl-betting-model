"""Neo4j graph store for NFL teams and the people associated with them.

Graph model
-----------
Nodes:
    (:Team   {abbr, name, nick, conf, division, team_id})
    (:Player {gsis_id, name, position, position_group, birth_date, college,
              height, weight})
    (:Coach  {name})
    (:Owner  {name, role})

Relationships:
    (:Player)-[:PLAYED_FOR {season, position, jersey_number, status}]->(:Team)
    (:Coach) -[:COACHED   {season, role}]->(:Team)
    (:Owner) -[:OWNS]->(:Team)
"""

from __future__ import annotations

import os
from collections.abc import Iterable

from neo4j import GraphDatabase

DEFAULT_URI = "bolt://localhost:7687"
DEFAULT_USER = "neo4j"
DEFAULT_PASSWORD = "password"

# Batch size for UNWIND-based writes.
_BATCH = 1000


def _clean(value):
    """Normalize pandas/NaN values into Neo4j-safe scalars."""
    import math

    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _records(rows: Iterable[dict]) -> list[dict]:
    return [{k: _clean(v) for k, v in row.items()} for row in rows]


class GraphStore:
    """Thin wrapper around the Neo4j driver with idempotent ingestion."""

    def __init__(self, uri: str | None = None, user: str | None = None,
                 password: str | None = None):
        self.uri = uri or os.getenv("NEO4J_URI", DEFAULT_URI)
        self.user = user or os.getenv("NEO4J_USER", DEFAULT_USER)
        self.password = password or os.getenv("NEO4J_PASSWORD", DEFAULT_PASSWORD)
        self._driver = GraphDatabase.driver(
            self.uri, auth=(self.user, self.password)
        )

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> "GraphStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def verify(self) -> None:
        """Raise if the database is unreachable."""
        self._driver.verify_connectivity()

    def _write(self, cypher: str, rows: list[dict]) -> None:
        with self._driver.session() as session:
            for i in range(0, len(rows), _BATCH):
                batch = rows[i : i + _BATCH]
                session.run(cypher, rows=batch)

    # --- schema ----------------------------------------------------------

    def setup_constraints(self) -> None:
        stmts = [
            "CREATE CONSTRAINT team_abbr IF NOT EXISTS "
            "FOR (t:Team) REQUIRE t.abbr IS UNIQUE",
            "CREATE CONSTRAINT player_gsis IF NOT EXISTS "
            "FOR (p:Player) REQUIRE p.gsis_id IS UNIQUE",
            "CREATE CONSTRAINT coach_name IF NOT EXISTS "
            "FOR (c:Coach) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT owner_name IF NOT EXISTS "
            "FOR (o:Owner) REQUIRE o.name IS UNIQUE",
        ]
        with self._driver.session() as session:
            for stmt in stmts:
                session.run(stmt)

    # --- ingestion -------------------------------------------------------

    def ingest_teams(self, rows: Iterable[dict]) -> None:
        cypher = """
        UNWIND $rows AS row
        MERGE (t:Team {abbr: row.abbr})
        SET t.name = row.name, t.nick = row.nick, t.conf = row.conf,
            t.division = row.division, t.team_id = row.team_id
        """
        self._write(cypher, _records(rows))

    def ingest_players(self, rows: Iterable[dict]) -> None:
        cypher = """
        UNWIND $rows AS row
        MERGE (p:Player {gsis_id: row.gsis_id})
        SET p.name = row.name, p.position = row.position,
            p.position_group = row.position_group, p.birth_date = row.birth_date,
            p.college = row.college, p.height = row.height, p.weight = row.weight
        """
        self._write(cypher, _records(rows))

    def ingest_roster_edges(self, rows: Iterable[dict]) -> None:
        cypher = """
        UNWIND $rows AS row
        MATCH (p:Player {gsis_id: row.gsis_id})
        MATCH (t:Team {abbr: row.team})
        MERGE (p)-[r:PLAYED_FOR {season: row.season}]->(t)
        SET r.position = row.position, r.jersey_number = row.jersey_number,
            r.status = row.status
        """
        self._write(cypher, _records(rows))

    def ingest_coaches(self, rows: Iterable[dict]) -> None:
        cypher = """
        UNWIND $rows AS row
        MERGE (c:Coach {name: row.name})
        WITH c, row
        MATCH (t:Team {abbr: row.team})
        MERGE (c)-[r:COACHED {season: row.season}]->(t)
        SET r.role = row.role
        """
        self._write(cypher, _records(rows))

    def ingest_owners(self, rows: Iterable[dict]) -> None:
        cypher = """
        UNWIND $rows AS row
        MERGE (o:Owner {name: row.name})
        SET o.role = row.role
        WITH o, row
        MATCH (t:Team {abbr: row.team})
        MERGE (o)-[:OWNS]->(t)
        """
        self._write(cypher, _records(rows))

    # --- inspection ------------------------------------------------------

    def counts(self) -> dict[str, int]:
        """Return node/relationship counts for a quick sanity check."""
        queries = {
            "Team": "MATCH (t:Team) RETURN count(t) AS n",
            "Player": "MATCH (p:Player) RETURN count(p) AS n",
            "Coach": "MATCH (c:Coach) RETURN count(c) AS n",
            "Owner": "MATCH (o:Owner) RETURN count(o) AS n",
            "PLAYED_FOR": "MATCH ()-[r:PLAYED_FOR]->() RETURN count(r) AS n",
            "COACHED": "MATCH ()-[r:COACHED]->() RETURN count(r) AS n",
            "OWNS": "MATCH ()-[r:OWNS]->() RETURN count(r) AS n",
        }
        out: dict[str, int] = {}
        with self._driver.session() as session:
            for label, q in queries.items():
                out[label] = session.run(q).single()["n"]
        return out
