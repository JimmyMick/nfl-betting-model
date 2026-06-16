# RATING.md — How the model rates teams and produces win probabilities

This document traces the whole pipeline end to end: from raw data load, through
the rating signals and the calibrated probability model, to evaluation and the
weekly reporting surfaces.

**What this system is:** a calibrated win-probability forecaster for NFL games,
benchmarked at every step against the Vegas moneyline. After an honest betting
evaluation (see [§7](#7-evaluation)), the moneyline market proved efficient —
the model cannot beat the closing line for profit — so the project is positioned
as a **probability / preview tool**, not a betting tip sheet. No picks, no EV
claims in the reporting layer.

```
nflreadpy ─┬─ schedules ──────────────► data.load_games ─► home_win target
           ├─ play-by-play ─► epa.team_game_epa ─┐
           ├─ play-by-play ─► qb.starting_qb_ovr ┤
           └─ snap counts  ─► starters.starter_unit_ovr
                                                 │
   Madden ratings (theedgepredictor) ─► madden ─┘
                                                 │
                         elo.compute_elo ────────┤
                                                 ▼
                                      features.build_features
                                   (leak-free, home-minus-away diffs)
                                                 │
                                                 ▼
                                     model.train (logistic | gbm,
                                  time-aware isotonic/sigmoid calibration)
                                                 │
              ┌──────────────────────────────────┼───────────────────────────┐
              ▼                                   ▼                           ▼
   model.evaluate / main.py            predict.py (weekly preview)   grade.py (grading +
   (ablation, walk-forward,            dashboard.py (3 tabs)         season tracker)
    betting ROI vs market)             weeks.py (--auto detection)
```

---

## 1. Data load (`nfl_betting_model/data.py`)

`load_games(seasons, include_unplayed=False)` pulls NFL schedules via
`nflreadpy` (polars → pandas), keeps a fixed column set (teams, scores, rest,
moneylines, spread, roof/location, division flag), and shapes them for modeling:

- **Chronological, deterministic order.** Rows are sorted by `(gameday,
  game_id)`. The `game_id` tiebreaker matters: same-day games would otherwise
  keep nflreadpy's load order, which varies run-to-run and leaks into the
  calibration step's tie handling (see [§9](#9-reproducibility)).
- **Target.** `home_win` = 1 if the home team scored more, else 0. Played ties
  (rare) are dropped — undefined for a binary winner model.
- **Unplayed games.** With `include_unplayed=True`, scheduled-but-unplayed games
  are kept with `home_win = NaN`. The training path leaves this `False` (so it
  only ever sees completed games); the weekly inference path turns it on to score
  an upcoming slate.

`to_long(games)` explodes each game into two team-perspective rows (one per team,
with points for/against and `won`). This is the substrate the rolling-form
features are built on.

---

## 2. Rating signals (the inputs)

The model does not use a single "rating." It combines several pre-game signals,
each computed so a game never sees its own result. Every signal is ultimately
expressed as a **home-minus-away difference** in [§4](#4-feature-assembly).

### 2a. Recent form (`features.py`)
Per-team rolling means over the last `FORM_WINDOW = 5` games, always `shift(1)`'d
so the current game is excluded:
- `form_pf` / `form_pa` — points for / against
- `form_margin` — `form_pf − form_pa`
- `form_winrate` — rolling win rate
- `season_winrate` — expanding, within-season win rate (also shifted)
- plus `rest_diff` (days rest, home − away) and `div_game`.

### 2b. Elo ratings (`elo.py`)
A FiveThirtyEight-style Elo, producing strictly pre-game ratings:
- Start `BASE = 1500`; **home-field advantage** `HFA = 55` Elo points folded into
  the home side before computing the expected result.
- Expected home win prob = `1 / (1 + 10^(-(elo_home+HFA − elo_away)/400))`.
- After each *played* game, ratings update by `K · MOV · (result − expected)`
  with `K = 20`. The **margin-of-victory multiplier** scales by
  `log(|margin|+1)` and damps the favorite-autocorrelation via the winner's Elo
  gap (so blowouts by already-strong teams move ratings less).
- **Season reversion:** at each team's first game of a new season, the rating
  reverts `REVERT = 0.33` of the way back toward 1500.
- **Unplayed games** emit pre-game Elo and win prob but do **not** update ratings.

Exposes `elo_diff` (home edge incl. HFA) and `elo_prob` (Elo's own win prob).

### 2c. EPA efficiency (`epa.py`)
From play-by-play (loaded one season at a time to bound memory):
- `off_epa` = mean EPA per play the team ran;
- `def_epa` = mean EPA per play the team allowed (lower is better).

These are *raw game outcomes*, so the feature layer rolls them through the same
leak-free `shift(1)` window as recent form before use.

### 2d. Madden starting-QB rating (`qb.py` + `madden.py`)
- The de-facto starter is the passer with the most dropbacks for a team in that
  game (from the game's own pbp).
- Joined to that season's **Madden launch OVR** on `(gsis_id, season)` from the
  theedgepredictor/nfl-madden-data dataset (cached under `data/madden/`).
- The QB's OVR is a **pre-season fixed** rating, so even though the starter is
  identified from in-game data, the rating value carries no in-game outcome
  (and starters are announced ~90 min pre-kickoff — realistic, not leakage).

### 2e. Madden starting-unit ratings (`starters.py`)
From nflverse **snap counts** (who actually played, and how much) joined to
Madden by `pfr_id`. "Starters" = season players above a **50% snap-share**
threshold. Aggregated into per-unit average OVR:
- `ol_ovr` (C/G/T), `dl_ovr` (DE/DT/NT), `db_ovr` (CB/FS/SS), and `starter_ovr`
  (all starters, both sides). Snap counts begin in 2012; earlier seasons get NaN.

### 2f. Starter availability (`availability.py`)
The one signal orthogonal to team strength: Elo/EPA rate a team's baseline
roster but are blind to a key starter — above all the QB — being ruled out *this
week*. From the official injury report (`load_injuries`, `report_status`), for
each team-week we sum the **talent above replacement** of players listed
Out/Doubtful: `weight · max(0, MaddenOVR − 65)` (Out=1.0, Doubtful=0.75). Because
QBs carry the highest OVRs, a QB ruled out dominates the sum naturally. Exposes
`out_avail` per team. Leak-free: the report's game-status designation is
published days before kickoff, exactly the pre-game info a bettor holds.

### 2g. Market benchmark (`features.market_home_prob`)
The Vegas moneyline is the yardstick, not an input feature. American odds →
implied probability for each side, then the pair is renormalized to remove vig:
`market_home_prob = p_home / (p_home + p_away)`. Used everywhere the model is
scored against "the market."

---

## 3. Why these are leak-free

Every signal that derives from game outcomes (form, EPA, Elo) is shifted so the
current game is excluded; the Madden ratings are season-fixed launch values.
`build_features` then **drops** any game with no prior-form signal on either side
(`form_margin_diff` / `form_winrate_diff` NaN), so early-season games without
history don't pollute training.

---

## 4. Feature assembly (`features.build_features`)

Takes `games` plus the optional `epa_table`, `elo_table`, `qb_table`,
`starter_table` and returns `(features_df, feature_cols)`. It:

1. Builds per-team rolling form on the long frame, merging EPA in first so it
   gets rolled too.
2. Looks each team's form columns back up onto the home and away side of each
   game, then forms **home-minus-away diffs**:
   `form_pf_diff, form_pa_diff, form_margin_diff, form_winrate_diff,
   season_winrate_diff, rest_diff, div_game`.
3. Layers optional blocks when their tables are supplied:
   - **EPA:** `off_epa_diff, def_epa_diff, net_epa_diff` (net = each side's
     offense − defense-allowed, home edge over away).
   - **Elo:** `elo_diff, elo_prob`.
   - **Madden QB:** `qb_ovr_diff` (home starter OVR − away starter OVR).
   - **Starters:** `ol_ovr_diff, dl_ovr_diff, db_ovr_diff, starter_ovr_diff`.
   - **Availability:** `out_avail_diff` (home talent-out minus away).

`feature_cols` lists exactly the active columns, so callers can ablate feature
sets cleanly. The full set is 17 features.

---

## 5. The probability model (`model.py`)

`build_pipeline(kind)`:
- **`logistic`** — median impute → standardize → L2 logistic regression. Default
  for the preview because it gives saner tail probabilities.
- **`gbm`** — `HistGradientBoostingClassifier` (handles NaNs + interactions
  natively). Marginally better aggregate calibration but uglier tails.

`train(train_df, cols, kind, calibrate)`:
- Uncalibrated: fit the pipeline directly.
- **Calibrated (`isotonic` | `sigmoid`):** *time-aware* calibration. The base
  model is fit on every season **except the latest** in the training data; the
  calibrator (`CalibratedClassifierCV` over a `FrozenEstimator`) is fit on that
  held-out latest season. So the probability mapping is learned on out-of-sample
  predictions without touching the test season. The validated production setup is
  **sigmoid (Platt) calibrated**.

> **Why sigmoid, not isotonic.** A 2021–2025 walk-forward (`calibration_study.py`)
> found isotonic calibration on the single latest season overfits its ~285-game
> map and detonates when that season is anomalous (calibrating 2021 on the 2020
> COVID season pushed log loss to ~1.0). It averaged **0.728** log loss — worse
> than uncalibrated — with ~10× the season-to-season variance. Sigmoid (two
> parameters, stable on small samples) averages **0.624** and cuts variance ~10×.

---

## 6. Training discipline (leakage control at the season level)

`time_split(df, test_season)` trains on every season **strictly before**
`test_season` and tests on that season. The weekly/grading paths use the same
rule: a model that scores season *S* is trained only on seasons `< S`. Combined
with the within-season `shift(1)` features, no information from the predicted
game (or its season's future) reaches the model.

---

## 7. Evaluation

### 7a. Metrics (`model.evaluate`)
Per holdout season, for the model and for the market on the same games:
- **Accuracy** (straight-up pick correctness)
- **Log loss** and **Brier score** (calibration / probabilistic accuracy)
- **AUC** (ranking quality, model only)

### 7b. Ablation (`main.py`, default mode)
Adds one signal block at a time on a single holdout season and prints each
against the market: `base form → +Elo → +Elo+EPA → +QB → +QB+Starters`, across
logistic and gbm, including calibrated variants. This isolates each signal's
marginal contribution. Headline result on a held-out season: the full feature set
is the best probability model — its log loss and Brier **edge the market** — but
straight-up accuracy stays ~1 point under the market.

### 7c. Walk-forward backtest (`main.py --backtest START-END`)
Expanding-window backtest: each test season is predicted by a model trained on
all prior seasons, and results are pooled. Reports per-season accuracy vs market
and pooled betting ROI.

### 7d. Betting ROI vs the closing line (`betting.py`)
The honest test of whether calibrated probabilities translate into money:
- Convert each side's American odds to a payout multiplier and to the
  vig-included implied price.
- For each game, take the **higher-EV side** and bet a flat 1 unit only if its
  expected value clears a threshold (`EV ≥ 0%, 2%, 5%, 10%`).
- Report bets placed, win rate, profit, and ROI; `combine` pools across seasons.

**Finding:** across a multi-season walk-forward (2019–2025, ~1,600+ bets) every
configuration loses money. Calibration roughly halves the loss but creates no
alpha; tightening the EV threshold *lowers* the win rate (the flagged "edges" are
illusory). Notably, the season with the **best calibration** had among the
**worst ROI** — proof that calibration ≠ a betting edge when the market is
equally calibrated and vig is in the price. **Conclusion: the moneyline market is
efficient for this feature set.** Hence the pivot to a probability/preview tool.

---

## 8. Reporting & inference surfaces

### 8a. Weekly preview (`predict.py`)
`predict_week(season, week, …)`:
1. `_prepare_frame` loads all seasons up to the target, builds Elo/EPA/QB/starter
   tables (pbp tables only on seasons with games played), and **carries forward**
   each team's latest known starter OVR onto a not-yet-played slate (a no-op for
   historical weeks).
2. `_train_for` fits the sigmoid-calibrated model on every season before the
   target.
3. Scores the slate, computes `edge = model_home_prob − market_home_prob`, and
   names a **key driver** per game — the largest z-scored feature diff whose sign
   agrees with the model's lean (interpretable "why").

`render` emits a markdown table sorted by disagreement size, plus the biggest
model-vs-market gaps and (on graded weeks) straight-up accuracy vs market.

### 8b. Grading & season tracker (`grade.py`)
After a week is final, `grade_season` scores every completed game in the season
(same train-once-per-season setup) and reports:
- straight-up record + accuracy, model vs market;
- calibration (log loss / Brier) vs market;
- this week's game-by-game ✓/✗;
- a week-by-week table with a cumulative season row.

### 8c. Schedule-aware automation (`weeks.py`)
`detect_target(mode)` reads the live schedule so scheduled runs need no
hard-coded week: `preview` returns the earliest unplayed week **once its first
game is within 5 days** (so the Thursday job starts at Week 1, not a week early);
`grade` returns the most recent completed week. Off-season it raises `SystemExit`
so the job stays silent. `predict.py --auto` / `grade.py --auto` use this; two
cron jobs (Thursday preview, Tuesday grade) post to chat in-season.

### 8d. Dashboard (`dashboard.py`)
A Streamlit shell with shared season/model/train-start controls and three tabs:
- **Weekly preview** — the model-vs-market table, biggest disagreements, edge
  chart.
- **Season tracker** — season-to-date record + calibration vs market, a
  cumulative accuracy ticker, week-by-week table, and latest-week grades.
- **Team roster** (`roster.py`) — a team's players with Madden ratings,
  snap-share starter flags, starter talent by unit, and a ratings distribution.

Results are cached per slate/model so re-runs are instant.

---

## 9. Reproducibility

The pipeline is deterministic up to a negligible residual. The `(gameday,
game_id)` sort in `load_games` removed the main source of run-to-run drift (~1%
probability wobble that leaked through isotonic tie-handling). A tiny residual
(~0.0016) remains from polars' multithreaded float aggregation in the pbp
modules; it never changes a displayed percentage or a pick, so it is accepted
rather than forced single-threaded.

---

## 10. File map

| File | Role |
|---|---|
| `nfl_betting_model/data.py` | Load schedules, build `home_win`, deterministic sort |
| `nfl_betting_model/elo.py` | 538-style pre-game Elo ratings |
| `nfl_betting_model/epa.py` | Per-team-game offensive/defensive EPA from pbp |
| `nfl_betting_model/madden.py` | Load Madden player ratings (by gsis_id / pfr_id) |
| `nfl_betting_model/qb.py` | Starting-QB Madden OVR per game |
| `nfl_betting_model/starters.py` | Starting-unit (OL/DL/DB) Madden OVR per game |
| `nfl_betting_model/availability.py` | Injury-report talent-out (availability) per team-game |
| `nfl_betting_model/features.py` | Leak-free feature assembly + market benchmark |
| `nfl_betting_model/model.py` | Pipelines, time-aware calibration, evaluation |
| `nfl_betting_model/betting.py` | +EV flat-stake ROI vs the closing line |
| `nfl_betting_model/weeks.py` | Schedule-aware target-week detection |
| `nfl_betting_model/roster.py` | Team roster: snaps + Madden ratings |
| `main.py` | Ablation + walk-forward backtest harness |
| `predict.py` | Weekly model-vs-market preview |
| `grade.py` | Weekly grading + season-to-date tracker |
| `dashboard.py` | Streamlit dashboard (preview / tracker / roster) |
