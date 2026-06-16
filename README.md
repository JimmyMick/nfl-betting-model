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

### Weekly preview (CLI)

```bash
uv run predict.py --season 2026 --week 1 --out predictions/2026-wk01.md
```

Trains the isotonic-calibrated full-feature model on every season before the
target, then predicts the slate from strictly pre-game features and writes a
model-vs-market markdown table (no picks, no EV claims).

### Grading / season tracker (CLI)

```bash
uv run grade.py --season 2024 --week 10 --out predictions/2024-grade-wk10.md
```

After a week is final, grades the model's straight-up picks (✓/✗) and tracks
season-to-date accuracy and calibration (log loss / Brier) against the market.
The Tuesday companion to the Thursday preview.

### Dashboard

```bash
uv run streamlit run dashboard.py
```

Interactive shell over the weekly-preview pipeline: pick a season/week/model in
the sidebar and browse the model-vs-market table, the biggest disagreements, an
edge chart, and (for graded weeks) straight-up accuracy vs Vegas. The first run
for a given slate trains the model (~30–60s); results are cached per
slate+model.

## How it works

- **Data** (`nfl_betting_model/data.py`) — loads completed games via `nflreadpy`
  and builds the `home_win` target.
- **Features** (`nfl_betting_model/features.py`) — strictly pre-game signals with
  no leakage (every stat is `shift(1)`'d so a game never sees its own result):
  rolling 5-game form (points for/against, margin, win rate), season-to-date win
  rate, rest-day difference, divisional flag. Composable Elo and EPA blocks layer
  on optionally. Also derives vig-free implied probabilities from the moneyline.
- **Elo** (`nfl_betting_model/elo.py`) — 538-style ratings with home-field
  advantage, margin-of-victory scaling, and between-season reversion to the mean.
  Ratings are read before each game and updated after, so the `elo_diff` /
  `elo_prob` features never leak.
- **EPA** (`nfl_betting_model/epa.py`) — per-team offensive and defensive
  expected-points-added per play from `load_pbp`, aggregated one season at a time
  and turned into leak-free rolling form by the feature layer.
- **Model** (`nfl_betting_model/model.py`) — either `SimpleImputer` →
  `StandardScaler` → `LogisticRegression`, or a `HistGradientBoostingClassifier`
  (handles NaNs + interactions). Time-based split, evaluated against the market
  (accuracy, log loss, Brier, AUC). `main.py` runs an ablation across feature
  sets and both model types.

## Ablation (trained 2010–2022, tested on 2023)

| Config | Accuracy | Log loss | Brier | AUC |
|---|---|---|---|---|
| base form (logistic) | 61.8% | 0.657 | 0.233 | 0.637 |
| + Elo (logistic) | 61.4% | **0.650** | **0.229** | **0.659** |
| + Elo + EPA (logistic) | 61.8% | 0.655 | 0.231 | 0.651 |
| + Elo + EPA (GBM) | **64.9%** | 0.661 | 0.233 | 0.642 |
| Vegas market | 67.0% | 0.627 | 0.219 | — |

Takeaways: **Elo gives the best calibration** (log loss / Brier / AUC); **EPA is
collinear with Elo** and barely moves the linear model, but the **gradient-boosted
model turns it into accuracy** (61.8% → 64.9%, closing ~1/3 of the gap to the
market). The model learns real signal but still does not beat the closing line —
the standing benchmark to chase.

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

- QB-adjusted Elo (rating travels with the starting quarterback)
- Richer EPA splits (pass vs rush, success rate, early-down EPA, opponent-adjusted)
- ROI / betting-edge evaluation vs closing lines (the metric that actually pays)
- Link the graph into the model (coach tenure, roster continuity as features)
- Hyperparameter tuning + walk-forward (multi-season) backtesting
