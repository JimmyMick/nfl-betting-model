# nfl-betting-model

NFL straight-up (moneyline) win-probability model.

Predicts the probability that the **home team wins** a given game, using only
information available before kickoff, and benchmarks itself against the Vegas
moneyline.

## Setup

```bash
uv sync
```

## Usage

```bash
uv run main.py --train 2010-2022 --test 2023
```

`--train` takes an inclusive season range (`2010-2022`); `--test` takes a single
held-out season.

## How it works

- **Data** (`nfl_betting_model/data.py`) — loads completed games via `nflreadpy`
  and builds the `home_win` target.
- **Features** (`nfl_betting_model/features.py`) — strictly pre-game signals with
  no leakage (every stat is `shift(1)`'d so a game never sees its own result):
  rolling 5-game form (points for/against, margin, win rate), season-to-date win
  rate, rest-day difference, divisional flag. Also derives vig-free implied
  probabilities from the moneyline for benchmarking.
- **Model** (`nfl_betting_model/model.py`) — `SimpleImputer` →  `StandardScaler` →
  `LogisticRegression`, with a time-based train/test split and evaluation against
  the market (accuracy, log loss, Brier, AUC).

## v1 baseline (trained 2010–2022, tested on 2023)

| | Accuracy | Log loss | Brier | AUC |
|---|---|---|---|---|
| Model | 61.8% | 0.657 | 0.233 | 0.637 |
| Vegas market | 67.0% | 0.627 | 0.219 | — |

The model learns real signal (beats the ~55% always-pick-home baseline) but does
not beat the market yet.

## Graph store (Neo4j)

Teams and the people associated with them are modeled as a graph.

**Nodes:** `Team`, `Player`, `Coach`, `Owner`
**Relationships:**
- `(:Player)-[:PLAYED_FOR {season, position, jersey_number, status}]->(:Team)`
- `(:Coach)-[:COACHED {season, role}]->(:Team)`
- `(:Owner)-[:OWNS]->(:Team)`

Data sources: teams/rosters/coaches come from `nflreadpy` (coaches via the
schedule's `home_coach`/`away_coach`). **Ownership has no `nflreadpy` feed**, so
it is loaded from `data/owners.csv` — a user-maintained file; verify it for the
current season, as ownership changes hands.

Connection is configured via env vars (`NEO4J_URI`, `NEO4J_USER`,
`NEO4J_PASSWORD`); see `.env.example`.

## Running with Docker

`docker-compose.yml` runs Neo4j plus the app image. The app waits for Neo4j's
health check before connecting.

```bash
cp .env.example .env          # set NEO4J_PASSWORD (min 8 chars)
docker compose up -d neo4j    # start the database
docker compose run --rm app   # build + ingest the graph (default command)
```

Then browse the graph at <http://localhost:7474> (bolt on `7687`).

Run the betting model in the same image:

```bash
docker compose run --rm app uv run main.py --train 2010-2022 --test 2023
```

Or run either script directly on the host (deps via `uv sync`):

```bash
uv run ingest_graph.py --seasons 2022-2023
uv run query_demo.py          # sample relationship queries
```

## Next steps

- Elo team ratings
- EPA / play-level features from `nflreadpy.load_pbp`
- Gradient boosting (`HistGradientBoostingClassifier`)
- Probability calibration + ROI evaluation vs closing lines
- Link the graph into the model (coach tenure, roster continuity as features)
