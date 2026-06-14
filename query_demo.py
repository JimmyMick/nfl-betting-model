"""Sample Cypher queries against the NFL graph — a quick smoke test / demo.

    uv run query_demo.py
"""

from __future__ import annotations

from nfl_betting_model.graph import GraphStore

TEAM_PROFILE = """
MATCH (t:Team {abbr: $abbr})
OPTIONAL MATCH (o:Owner)-[:OWNS]->(t)
OPTIONAL MATCH (c:Coach)-[:COACHED {season: $season}]->(t)
OPTIONAL MATCH (p:Player)-[r:PLAYED_FOR {season: $season}]->(t)
    WHERE r.position = 'QB'
RETURN t.name AS team, o.name AS owner, c.name AS coach,
       collect(DISTINCT p.name)[..5] AS quarterbacks
"""

COACH_MOVES = """
MATCH (c:Coach)-[r:COACHED]->(t:Team)
WITH c, collect(DISTINCT t.abbr) AS teams
WHERE size(teams) > 1
RETURN c.name AS coach, teams
ORDER BY coach
LIMIT 8
"""


def main() -> None:
    with GraphStore() as g:
        g.verify()
        with g._driver.session() as s:
            print("=== Team profile: KC, 2023 ===")
            rec = s.run(TEAM_PROFILE, abbr="KC", season=2023).single()
            for k in ("team", "owner", "coach", "quarterbacks"):
                print(f"  {k:13s}: {rec[k]}")

            print("\n=== Coaches with >1 team in the data ===")
            for rec in s.run(COACH_MOVES):
                print(f"  {rec['coach']:22s} {rec['teams']}")


if __name__ == "__main__":
    main()
