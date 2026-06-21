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
model-vs-market markdown table (no picks, no EV claims). Pass `--auto` instead of
`--season/--week` to target the upcoming slate detected from the live schedule
(used by the scheduled Thursday preview).

### Grading / season tracker (CLI)

```bash
uv run grade.py --season 2024 --week 10 --out predictions/2024-grade-wk10.md
```

After a week is final, grades the model's straight-up picks (✓/✗) and tracks
season-to-date accuracy and calibration (log loss / Brier) against the market.
The Tuesday companion to the Thursday preview. `--auto` targets the most recent
completed week from the live schedule.

### Expert pick'em tracker (CLI)

Track your own picks — and your friends' — against the model, on the same games,
all season. Each participant picks a **winner** and a **confidence** (50–100) per
game; the grader scores everyone on straight-up accuracy plus calibration
(Brier / log loss), exactly the footing the model gets.

```bash
# 1. List participants — one name per line — in predictions/picks/players.txt
# 2. Seed a blank sheet for the week (auto-filled from that week's schedule):
uv run picks.py --season 2026 --week 1        # or --auto for the upcoming slate
# 3. Fill in the 'pick' (team abbrev) and 'confidence' columns, then grade:
uv run grade.py --season 2026 --week 1
```

Picks live in `predictions/picks/{season}-wk{NN}.csv` (one row per game × player,
git-tracked as the season's record). Leaving a `confidence` blank still counts a
pick toward the win/loss record — it's just excluded from the Brier / log-loss
calibration columns. The leaderboard prints automatically at the bottom of every
`grade.py` run whenever picks exist, and has its own dashboard tab.

### Dashboard

An interactive Streamlit shell over the whole pipeline — the preview, grading,
pick'em, and roster views without touching the CLI.

**Launch it:**

```bash
uv run streamlit run dashboard.py
```

This opens the app at <http://localhost:8501> (Streamlit usually opens your
browser automatically; if not, visit that URL). To run it without auto-opening a
browser — e.g. on a remote host — add `--server.headless true`. Stop the server
with `Ctrl-C` in the terminal.

**Sidebar controls** (shared across all tabs):

- **Season** — which NFL season to load.
- **Week** — the slate to preview (Weekly preview tab).
- **Model** — `logistic` (default; saner tail probabilities) or `gbm`
  (marginally better aggregate calibration, uglier tails).
- **Train start** — the first season used for training; the model always trains
  on every season from here up to (but not including) the one being scored.

Change any control and the affected tab recomputes. The first run for a given
slate/model trains the model (~30–60s); results are then **cached**, so
switching tabs or revisiting a slate is instant.

**Tabs:**

- **Weekly preview** — pick a season + week and click **Run preview**: the
  model-vs-market table, the biggest disagreements as metric cards, and an edge
  bar chart. A "what does the model see that the market doesn't" view — no picks,
  no bet sizing. Works on upcoming 2026 weeks via carry-forward of each team's
  latest starter ratings.
- **Season tracker** — pick a completed week and click **Run tracker** for the
  season-to-date straight-up record and calibration (log loss / Brier) vs the
  market, a cumulative accuracy ticker chart, the week-by-week table, and the
  latest completed week's game-by-game ✓/✗ grades.
- **Pick'em leaderboard** — pick a completed week and click **Run leaderboard**
  for the season standings of you + your friends vs the model: each player's
  record, their accuracy minus the model's *on the games they picked* ("vs
  Model"), Brier / log loss, an accuracy bar chart, and the latest week's
  game-by-game head-to-head. Reads the `predictions/picks/*.csv` sheets; shows a
  friendly prompt if none are filled yet.
- **Team roster** — pick a team and click **Show roster** for its players with
  Madden ratings, snap-share-based starter flags, starter talent by unit, and a
  ratings distribution. **Note:** this tab needs snap-count data, which only
  exists once a season is underway — choose a past season (e.g. 2024); a
  not-yet-started season like 2026 will show a friendly "no data yet" message.

#### Reading the Weekly preview

The tab has four parts, top to bottom:

1. **Biggest model-vs-market disagreements** — the three games where the model
   differs most from the closing line, each shown as a card:
   `BUF @ HOU / BUF +19% / roster talent → BUF`. Read it as matchup, edge, key
   driver (all three defined below).
2. **Straight-up accuracy** (graded/past weeks only) — the model's win-pick
   accuracy vs the market's, on games already played.
3. **Slate** — every game, **sorted by model win probability** (most confident
   first, coin-flips last). Columns:

   | Column | Meaning |
   |---|---|
   | **Matchup** | `AWAY @ HOME` — away team first, home team second. |
   | **Model** | The model's favoured side + its win probability, e.g. `BUF 74%`. |
   | **Market** | The same, implied by the closing moneyline, e.g. `BUF 55%`. |
   | **Edge** | How much *more* the model likes its side than the market does (model minus market), on the favoured side, e.g. `BUF +19%`. A **disagreement** measure, not a margin of victory and not a bet. |
   | **Key driver** | The single factor pushing the model toward its pick hardest (see table below), e.g. `roster talent → BUF`. |
   | **Result** | On graded weeks only: the actual winner + ✓/✗ for the model's pick. |

4. **Edge by game** — the same edges as a bar chart, longest disagreement on top.

   A **⬇ Download preview as PDF** button exports the summary + sorted slate as a
   one-page landscape report.

**Key-driver vocabulary** — the driver names the largest standardized pre-game
gap pointing toward the model's pick. It's an explanation aid, not the whole
model (which combines all signals):

| Shown as | Underlying signal |
|---|---|
| QB rating | starting-QB Madden overall gap |
| roster talent | starter OL/DL/DB overall gap |
| net EPA/play | offense-minus-defense efficiency gap |
| Elo | overall team-strength rating gap |
| recent margin | recent scoring-margin form |
| injury availability | talent ruled out (inactives) gap |
| model interactions | no single gap dominates; the combination drove the pick |

> **A big edge is not a betting signal.** It means the model disagrees with an
> efficient market — which is usually the model being wrong, not the market.
> Five feature experiments confirmed the moneyline is not beatable here; this is
> a calibrated *forecaster*, not a tip sheet.

#### Reading the Season tracker

- **Model vs Market straight-up** — win-pick records and the gap between them.
  Expect the model to land within ~1–3 points of the market, not beat it.
- **Calibration (log loss / Brier)** — lower is better; "market-grade" means the
  two are within a few thousandths. This is the bar the model actually clears.
- **Accuracy ticker** — cumulative pick accuracy by week, model vs market.
- **Week-by-week table** + **latest-week ✓/✗ grades** — the game-level detail.

#### Reading the Pick'em leaderboard

- **Standings** — players ranked by straight-up accuracy, with their record and
  **vs Model** (their accuracy minus the model's over the *same* games they
  picked). A positive "vs Model" means they're beating the forecaster on their
  own slate — the headline number.
- **Brier / Log loss** — calibration of each player's confidences (lower is
  better), comparable to the model's own numbers in the Season tracker. Computed
  only over picks that carried a confidence, so a player who skips confidences
  still gets a win/loss record but no calibration score.
- **Accuracy bar** + **game-by-game head-to-head** — who's hot, and exactly which
  games each person nailed or missed in the latest week.

#### Reading the Team roster

- **Starter talent by unit** — average Madden overall for QB / Offense / Defense
  / all starters; these are the same signals the model's `roster talent` and
  `QB rating` drivers come from.
- **Ratings distribution** — each player as a dot by overall rating and unit,
  starters highlighted.
- **Roster table** — per-player position, snap share, starter flag (season-avg
  snap share ≥ 50% on offense or defense), and key Madden attributes.

### Cloud dashboard (Streamlit Community Cloud)

`dashboard.py` trains the model live, which is too heavy for the free tier's
~1 GB. So the cloud app (`streamlit_app.py`) is a **read-only viewer**: the local
weekly runs export their results as small CSVs under `predictions/cloud/`, those
get committed + pushed, and the deployed app just renders them — no training, no
`nflreadpy` fetch, no Madden data needed in the cloud. Its dependencies are the
light set in `requirements.txt` (streamlit / pandas / numpy / sklearn / altair),
separate from the full pipeline stack in `pyproject.toml`.

**Export the artifacts** (piggybacks on the training the weekly runs already do):

```bash
uv run grade.py   --season 2026 --week 5 --export-dir   # graded games + scored picks
uv run predict.py --season 2026 --week 6 --export-dir   # latest preview slate
```

`--export-dir` with no value writes to `predictions/cloud/`; pass a path to
override. This drops `graded_games.csv`, `scored_picks.csv`, `latest_preview.csv`
and a `meta.json` (season / week / timestamps). Commit and push them, and the
deployed app updates on its next load.

**Deploy:** push the repo to GitHub, then on [share.streamlit.io](https://share.streamlit.io)
create an app pointing at this repo with **`streamlit_app.py`** as the main file.
It installs `requirements.txt` and serves the three artifact-backed tabs
(Pick'em leaderboard, Season tracker, Weekly preview) — the tabs appear only once
their artifacts exist. Empty leaderboard until picks are recorded and a week is
graded.

## How it works

- **Data** (`nfl_betting_model/data.py`) — loads completed games via `nflreadpy`
  and builds the `home_win` target.
- **Features** (`nfl_betting_model/features.py`) — strictly pre-game signals with
  no leakage (every stat is `shift(1)`'d so a game never sees its own result):
  rolling 5-game form (points for/against, margin, win rate), season-to-date win
  rate, rest-day difference, divisional flag. Composable Elo, EPA, Madden
  QB/starter, and injury-**availability** blocks layer on optionally. Also derives
  vig-free implied probabilities from the moneyline.
- **Availability** (`nfl_betting_model/availability.py`) — the one signal
  orthogonal to team strength: per team-week, the Madden talent-above-replacement
  of players the injury report rules out (QBs dominate naturally). Leak-free
  (`report_status` is published pre-game).
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
  (accuracy, log loss, Brier, AUC). Probabilities are **sigmoid (Platt)
  calibrated** — isotonic-on-one-season overfit its calibration map and was ~10×
  noisier (see `calibration_study.py`). `main.py` runs an ablation across feature
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
