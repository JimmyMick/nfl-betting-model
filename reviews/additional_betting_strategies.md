# Additional Betting Strategies — NFL Rating System

> Based on thorough review of RATING.md §7 (Evaluation) and §2h (Documented Dead-Ends).
> Every strategy below is **not yet tested** in the current codebase and is designed
> to avoid re-treading the documented nulls.

---

## Review & Prioritization (2026-08, reviewed against the §7f evidence)

The guardrails here are excellent and match the project's discipline. But the
document is more optimistic than the evidence warrants: the moneyline market is
efficient for this feature set, and **only top-1 has ever cleared the honest
season-block bootstrap.** Reviewer's triage:

**Conceptual correction — A1 (Kelly).** The stated hypothesis ("could work when
flat betting fails") is backwards. Kelly sizes *positive*-EV bets; on a −EV bet it
sizes to zero. Sizing is a variance lever, not an edge source — it cannot rescue a
vig-bound strategy. And stake curves (flat/tiered/edge-proportional/Kelly/
confidence) were **already A/B-tested on the top-1 ledger** (during the paper-play
what-if work): normalized Kelly ≈ flat, edge-proportional best raw ROI but worst
drawdown, none better risk-adjusted. So A1/A2 are largely answered; only a
compounding-bankroll/risk-of-ruin treatment is untested, with a low ceiling.

**Structural weakness — N-power.** Top-1 is ~1 bet/week (~150 bets/7 seasons).
Every "slice top-1 by context" idea (B1 primetime, B2 rest, B4 division, D1
quadrant) cuts that to 20–40 bets → CIs too wide to conclude (cf. top-2 already
straddling zero at 297 bets).

**Likely already priced (orthogonality rule).** B2 (rest) and B4 (division) use
`rest_diff`/`div_game`, which are already model features the market prices; F2
overlaps `form_margin`. Near-certain nulls.

**Priority (mechanism orthogonal to pricing, adequate power, or new market):**

| Tier | Strategy | Status / rationale |
|---|---|---|
| **1** | **C1 availability-driven** | **TESTED → did NOT clear** (−4.1% ROI, season-block CI [−11.7%, +0.9%], P>0 10%). Availability is a real *calibration* signal but points mostly at favorites (avg ML −191); you win 62% but need ~66% — vig-bound. The market already prices injuries. See RATING.md §7f. |
| **2** | **E1 spreads** | Best remaining bet: the only structurally-different market (symmetric −110 ≈ 4.5% hold vs favorite ML juice 5–9%). Our "efficient" verdict is moneyline-only; a marginal edge could survive the lower vig. Needs a spread→prob calibration. |
| 3 | C2 Elo-only minimal model | Cheap; tests if washout features add noise to the betting tail. |
| 4 | B3 late-season filter | Cheap temporal split, distinct mechanism (Elo converged). |
| — | E2 CLV | Highest-value idea but **blocked on data** (no opening lines in nflreadpy). |
| skip | A1/A2 (answered), B1/B2/B4/D1 (underpowered/priced), D2/G1 (overfitting), E3/F1/F2 (weak prior / new data) | |

**Verdict:** C1 tested and failed (below). E1 (spreads) is the one genuinely new
frontier worth running next; the rest are low expected value against the
efficient-market prior and each costs ~7 training passes.

---

## Summary of What's Already Tested

| Strategy | Result |
|---|---|
| Flat EV ≥ threshold (0%, 2%, 5%, 10%) on higher-EV side | Every config loses money |
| Top-1/3/5 most-confident per week | ~80% accurate but no edge vs market |
| Top-1 biggest \|edge\| per week | **Clears bar** (CI +4%–+43%); paper-tracked |
| Top-2/3 biggest \|edge\| per week | Dilutes to break-even |
| Pick crossover (model picks different winner) | Carried by 2021; fails season-block bootstrap |
| Consensus favorite | −2.6% (vig-bound) |
| Model more confident than market on favorite | **−7.3% anti-predictive** |
| Fade model's overconfident favorites | +0.3% (dead break-even) |
| In-season weekly refit | Washes out with sigmoid calibration |
| Coaching, weather, PFF, per-player QB EPA, opponent-adjusted EPA | All null |

---

## Category A — Variable Stake Sizing

Everything tested so far uses flat 1-unit stakes. If the model's edge estimates
carry *any* informational content (and the top-1 disagreement result suggests
they do at extremes), then stake sizing is the obvious unexplored dimension.

### Strategy A1: Fractional Kelly Criterion

**The single biggest omission in the current test suite.**

**Hypothesis:** The model's calibrated probabilities are directionally informative
even when they can't beat a flat-EV threshold. Kelly (or fractional Kelly) converts
small probability edges into positive geometric growth by sizing bets proportionally
to edge magnitude, avoiding the all-or-nothing nature of flat betting.

**Mechanics:**
- For each game, compute the Kelly fraction: `f = (p_model × odds − (1−p_model)) / (odds − 1)`
- Test ¼-Kelly, ⅛-Kelly, and ¹⁄₁₆-Kelly (fractional Kelly is near-universally preferred
  in practice to damp estimation error)
- Cap individual stakes at, say, 5 units to prevent single-game blowup
- Compare geometric mean growth rate and max drawdown to the current flat-1 top-1 strategy

**Why this could work when flat betting fails:** A model probability of 58% on a +110
underdog has +15.8% EV per Kelly but only +5% raw edge — below the 10% flat threshold
but potentially profitable when sized correctly. Fractional Kelly also naturally
down-weights low-conviction bets that flat betting treats identically.

**Implementation notes:**
- Requires bankroll tracking across the full walk-forward window (not per-season reset)
- The season-block bootstrap must respect bankroll path-dependency (resample seasons,
  concatenate in order, run Kelly through the full sequence)
- Compare against flat-1 baseline on *risk-adjusted* return (Sharpe ratio) not just raw ROI
- Risk of ruin / max drawdown is the honest metric here, not raw ROI

**Expected challenge:** If the edge estimates are pure noise outside the top-1 per week,
Kelly just adds variance without helping. The test should report P(terminal bankroll > 1.0)
from the bootstrap, not just mean ROI.

---

### Strategy A2: Edge-Magnitude Proportional Staking

**Simpler than Kelly; tests whether edge magnitude is monotonic with value.**

**Hypothesis:** Bigger |edge| → higher win rate. If this holds, then a simple linear
stake rule (stake ∝ |edge|) should outperform flat-1.

**Mechanics:**
- Stake = `k × |edge|` units, where k normalizes to the same total exposure as flat-1
- Alternatively: tiered staking (1u for edge 2–5%, 2u for 5–10%, 3u for 10%+)
- Test both linear and tiered against flat-1 top-1

**Why this is worth testing separately from A1:** Kelly requires *calibrated* probabilities
to size correctly. Edge-proportional staking only requires *monotonic* edges — a weaker
condition. If the model's probabilities are well-calibrated in rank but not in magnitude,
this could outperform Kelly.

**Expected challenge:** The dilution finding (top-1 → top-2 → top-3 degrades fast)
suggests edge magnitude may not be monotonic beyond the single biggest disagreement.
This strategy might just confirm that finding with a different metric.

---

## Category B — Situational / Contextual Filters

The current strategies bet on *any* game that meets a numerical threshold. But the
model's edge may be concentrated in specific game contexts the market prices less
efficiently — primetime games with public betting pressure, rest-disparity mismatches,
or late-season when more data is available.

### Strategy B1: Primetime / Island Game Filter

**Hypothesis:** Primetime games (TNF, SNF, MNF, Thanksgiving, London/Germany) attract
disproportionate public betting volume. Sportsbooks shade lines toward public sentiment
(especially toward favorites and overs) to balance liability, creating small inefficiencies
a model not influenced by public bias can exploit.

**Mechanics:**
- Tag games by kickoff window: `primetime = True` for Thursday 8:20pm, Sunday 8:20pm,
  Monday 8:15pm, plus international games and Thanksgiving
- Run the top-1 disagreement strategy (the one that clears) but **only on primetime games**
  vs. **only on Sunday 1pm/4pm regional games**
- Compare ROI and season-block CIs between the two sets

**Test variant:** The opposite hypothesis — maybe primetime lines are *sharper* because
more attention means more efficient pricing. Test both directions.

**Implementation notes:**
- Requires game start-time data; nflreadpy schedules include `start_time`
- Primetime games are ~3-5 per week, so the N will be small — season-block bootstrap
  handles this honestly but CIs will be wide
- The Sunday 1pm window (~9-10 games/week) is the control

**Why this hasn't been tested:** The current system treats all games as fungible,
never slicing by context. The dead-end weather analysis did test a context split,
but primetime/public-betting is a different mechanism.

---

### Strategy B2: Rest Disparity Threshold

**Hypothesis:** Extreme rest disparities (Thursday game after OT, team off bye vs.
team on short rest) create physical mismatches that the moneyline may not fully price.
The model already has `rest_diff` as a feature, but it's never been tested as a
*betting filter* — only as an input.

**Mechanics:**
- Filter to games where `|rest_diff| ≥ 3` days (roughly top quartile of rest disparities)
- Run the top-1 disagreement strategy on this subset
- Also test: only bet the *well-rested* side of a disagreement (model likes the team
  with rest advantage), vs. only the tired side

**Rationale:** The rest_diff feature contributes to the model, but the key question is
whether the market systematically underweights rest. If the model's edge on rest-disparity
games is larger than on normal-rest games, this filter should improve ROI.

**Implementation notes:**
- Rest_diff is already in `features.py` and available pre-game
- Bye week → 13+ days rest; Thursday game → 3-4 days; normal → 6-7 days
- The extreme bins are small (maybe 2-4 games/week), so pool across multiple seasons for power

---

### Strategy B3: Late-Season Filter (Weeks 10–18)

**Hypothesis:** The model's Elo ratings and rolling-form features are noisiest early
in the season (small samples, offseason roster changes, rookie QBs). By Week 10, Elo
has converged, EPA is stable, and the model's disagreements with the market should be
sharper because both sides have more information.

**Mechanics:**
- Split every season into "early" (Weeks 1–9) and "late" (Weeks 10+)
- Run top-1 disagreement separately on each half
- Compare ROI and win rate

**Test variant:** The opposite — maybe early-season uncertainty creates *more*
mispricing as the market overreacts to small samples. This would manifest as better
ROI in Weeks 1–4.

**Why this hasn't been tested:** The backtests pool all weeks equally. A temporal
split tests whether the model's edge has a "learning curve" within each season —
which would inform practical deployment (only bet late-season disagreements).

---

### Strategy B4: Division vs. Non-Division Split

**Hypothesis:** Division games have higher familiarity (teams play twice/year),
tighter spreads, and more variance (rivalry effects). The model might perform
differently on division games because the features (Elo, form, EPA) capture
team quality but not matchup-specific familiarity.

**Mechanics:**
- `div_game` is already a feature in `features.py`
- Split the top-1 disagreement strategy into division and non-division subsets
- Hypothesis: non-division games should be more model-friendly (cleaner team-quality
  signal, less rivalry noise)

**Implementation notes:**
- Division games are ~6 per week out of ~16 total — decent sample
- If division games are significantly worse, the live paper play could filter them out

---

## Category C — Signal-Specific / Decomposition Strategies

The model combines 17 features into a single probability. But §2h shows most features
wash out individually, and only availability clearly adds value. Disaggregating *which
component* of the edge is driving the bet could separate signal from noise.

### Strategy C1: Availability-Only Bets

**Hypothesis:** Availability is the only feature that independently clears the
validation bar (5/5 logloss, 5/5 AUC). A strategy that only bets when the model-market
disagreement is **driven by availability** (not Elo or form) should be purer signal.

**Mechanics:**
- For each game, decompose the model's probability shift attributable to availability:
  predict with and without `out_avail_diff`, take the difference
- Within the top-N disagreements, only bet those where availability contributes ≥X%
  of the total edge (e.g., ≥33%)
- Alternatively: only bet top-1 disagreement when the **key driver** (§8a) is availability

**Test variants:**
- Only bet when the opposing starting QB is ruled Out/Doubtful (the strongest single
  availability signal)
- Only bet when `out_avail_diff` exceeds some absolute threshold (e.g., 5+ OVR points
  of talent ruled out)

**Why this could work when the full model's edge is noisy:** The top-1 disagreement
strategy works but is dilute — only 1 game/week. If availability-driven disagreements
are disproportionately the winners, you could get more bets without the dilution.

**Implementation notes:**
- The key driver is already computed in `predict.py` (largest z-scored feature diff
  whose sign agrees with model lean)
- Availability events are sparse (QB injuries), so N will be small — report CI honestly

---

### Strategy C2: Elo-Only Disagreement (Strip the Model Down)

**The reverse of C1 — test whether simplicity beats complexity.**

**Hypothesis:** Most features wash out in ablation. Maybe the full model's edge comes
entirely from Elo + HFA + availability, and the other features (EPA, QB OVR) add noise
that dilutes the disagreement signal. A minimal model (Elo + availability only) might
produce *fewer but sharper* disagreements.

**Mechanics:**
- Train a stripped model: Elo + HFA + availability + basic form only (drop EPA, QB OVR)
- Run top-1 disagreement on both models on the same games
- Compare ROI

**Why this hasn't been tested:** The ablation (§7b) tests marginal contribution for
*probability calibration* (log loss, Brier), not for *betting edge* specifically. A
feature that improves log loss could still hurt ROI if it adds noise to the tail
disagreements that drive bets.

---

## Category D — Confidence × Disagreement Interactions

The current top-1 strategy picks the single biggest |edge| per week. But §7f's
dogfade finding shows that model confidence and market confidence interact
non-linearly: the model is anti-predictive when it's *more confident than the
market* on favorites. This interaction hasn't been tested on the top-1 disagreement
framework.

### Strategy D1: Disagreement × Confidence Quadrant

**Hypothesis:** Not all disagreements are equal. A big |edge| driven by the model
being 80% confident when the market is 55% is different from a big |edge| where the
model is 58% and the market is 52%. The former might be the model correctly identifying
a mispricing; the latter might be noise around a coin flip.

**Mechanics (2×2 matrix):**
- **Quadrant I:** High |edge| + High model confidence (p ≥ 70% or ≤ 30%)
- **Quadrant II:** High |edge| + Low model confidence (p near 50%)
- **Quadrant III:** Low |edge| + High confidence
- **Quadrant IV:** Low |edge| + Low confidence (not bettable)

Run top-1 disagreement, classify each bet into I/II/III, and compare ROI across quadrants.

**Prediction:** Quadrant I (big edge, high confidence) should outperform Quadrant II
(big edge, low confidence). The current top-1 may include QII bets that are essentially
noise — filtering them out could improve ROI.

**Implementation notes:**
- Model confidence is already available: `max(p, 1−p)`
- This is a simple post-hoc filter on the existing top-1 strategy
- The N split will be uneven — report per-quadrant N honestly

---

### Strategy D2: The Confidence-Direction 2×2 (Extending the Dogfade Finding)

**Hypothesis:** §7f established that when the model is *more confident than the market*
on a favorite, it's anti-predictive (−7.3%). The reverse — model *less confident than
market* on an underdog — might also reverse sign. This creates a systematic 2×2.

**Mechanics:**
- For each game, compute `Δconf = model_confidence − market_confidence` on the side the
  model favors
- **Direction = Favorite** (model likes the chalk): if Δconf > 0 → FADE (bet the dog);
  if Δconf < 0 → FOLLOW (bet the favorite)
- **Direction = Underdog** (model likes the dog): if Δconf > 0 → FOLLOW (bet the dog);
  if Δconf < 0 → FADE (bet the favorite)

Then threshold on |edge| magnitude to select the best N/week.

**Why this is different from the existing dogfade test:** The dogfade study faded ALL
overconfident favorites indiscriminately. This 2×2 makes the fade *directional* —
only fade when confidence direction says to, and follow when it says to follow.

**Expected challenge:** This is getting into post-hoc territory — four cells, many
degrees of freedom. The season-block bootstrap is essential. Pre-register the hypothesis
before running the data.

---

## Category E — Market Structure / Alternative Bet Types

Everything tested so far is moneyline only. The model produces a win probability,
but it's never been tested on spreads or totals, or against opening lines.

### Strategy E1: Spread Betting with Model Disagreement

**Hypothesis:** The model's moneyline probability edge might translate to ATS (against
the spread) more profitably than moneyline, because spreads are tighter markets where
small probability advantages compound differently. The vig on spreads (−110 both sides,
~4.5% theoretical hold) is lower than the average moneyline vig on favorites.

**Mechanics:**
- Convert model win probability to an implied spread using a mapping (e.g., logistic
  regression of historical spread → win prob, or an SBR-style chart)
- When the model's implied spread differs from the market spread by ≥N points, bet
  the model's side ATS
- Test N = 1, 1.5, 2, 2.5, 3 points
- Run top-1/week and all-qualifying

**Why this could work:** The moneyline market efficiently prices win probability, but
spread betting involves a different set of participants (more recreational bettors,
different limits). The market might be efficient on the binary outcome but less so on
the margin of victory.

**Implementation notes:**
- Requires spread data; nflreadpy schedules include `spread_line`
- The spread-to-probability mapping needs calibration (don't use a simple linear model —
  the relationship is sigmoidal and varies by point total)
- Account for the −110 vig in ROI calculation

**Expected challenge:** If the moneyline is efficient, the spread probably is too —
they're the same underlying market. But the lower vig alone could flip marginal edges
from negative to positive.

---

### Strategy E2: Closing Line Value (CLV) — Bet Early, Measure vs. Close

**The gold-standard sports betting research question, never tested here.**

**Hypothesis:** The model's probability might predict *line movement*, not just closing-line
inefficiency. If the model's probability differs from the *opening* line in a direction
the line subsequently moves, that's genuine CLV — the holy grail of betting research.

**Mechanics:**
- Requires opening line data (Pinnacle or Bookmaker CRIS opens; nflreadpy may or may
  not have this — nflverse has some opening line data)
- For each game, compute: `model_direction` (does model like home or away more than
  opening line implies?) and `line_movement` (does the line move toward home or away
  from open → close?)
- Bet when model_direction == line_movement (model anticipated the steam)
- Compare ROI on "model was right about the move" vs. "model was wrong about the move"

**Implementation notes:**
- This is a major data dependency — opening lines may not be in the current pipeline
- If opening lines are unavailable, approximate with early-week market lines (Tuesday/Wednesday)
- This is a research project in itself, not a quick filter

**Why this is worth the effort:** If the model captures CLV, you don't need to beat
the closing line — you just need to bet before the line moves. This is how actual
professional betting groups operate.

---

### Strategy E3: Totals (Over/Under) Correlation Test

**Hypothesis:** Model confidence in the side might correlate with the total. A game
where the model is very confident about the favorite might tend to go *under* (mismatch
→ conservative 4th quarter) or *over* (mismatch → blowout garbage-time points).

**Mechanics:**
- For each game, record: model confidence (max(p, 1-p)), actual total points scored,
  market total line
- Test correlation between confidence and (actual_total − market_total)
- If significant, test a strategy: when top-1 disagreement + model confidence > threshold,
  bet the corresponding over/under direction

**Why this is speculative:** No strong prior on direction. This is exploratory —
worth a correlation check before building a strategy around it.

---

## Category F — Season-Level / Regime Strategies

### Strategy F1: Season Win-Total / Futures Implied Probability

**Hypothesis:** Preseason win totals and futures odds embed a different kind of market
wisdom than weekly moneylines. If the model disagrees with preseason expectations
systematically (e.g., the model was right about a team being undervalued all season),
betting that team weekly when the model likes them might compound.

**Mechanics:**
- Load preseason win totals or Super Bowl futures odds
- Tag each team as "market overrated" or "market underrated" based on model's
  preseason vs. market assessment
- Run top-1 disagreement, split by overrated/underrated tag
- Hypothesis: model edges on underrated teams should be more profitable than on
  overrated teams

**Implementation notes:**
- Preseason data source needed (Sportsbook Review archive, or nflreadpy)
- This is a lightweight filter on existing strategies

---

### Strategy F2: Post-Blowout Rebound / Letdown

**Hypothesis:** Teams coming off extreme outcomes (blowout win or blowout loss) may
be mispriced the following week. The market might overreact to a single extreme
performance (overvaluing a team that just dominated a bad opponent; undervaluing a
team that got crushed by a great one). The model's rolling 5-game form should be
more stable.

**Mechanics:**
- Tag games where one team is coming off a win by ≥17 points or a loss by ≥17
- Filter the top-1 disagreement strategy to only rebound/letdown games vs. all others
- Test both "bet on the team coming off a blowout loss" (rebound) and "bet against
  the team coming off a blowout win" (letdown)

**Implementation notes:**
- Previous-week margin is derivable from the game data already loaded
- Blowout threshold is a tunable parameter; sweep 14/17/21 points
- This is conceptually similar to the momentum-form features but tests whether the
  *market* overweights extreme single-game results

---

## Category G — Meta-Strategies

### Strategy G1: Anti-Jinx / Reverse Paper Play

**The contrarian double-check.**

**Hypothesis:** The dogfade finding (−7.3% when model overconfident on favorites)
suggests a systematic bias. What if the model's *worst* disagreements — where it's
most wrong historically — are themselves predictable and fade-able?

**Mechanics:**
- On the walk-forward, identify the characteristics of games where the top-1
  disagreement lost (model's biggest edge and it was wrong)
- Look for common patterns: was the losing bet always when the model liked a road
  favorite? When Elo and EPA disagreed? When availability was zero?
- If a pattern emerges, test: when top-1 disagreement matches the "likely to lose"
  profile, bet the opposite side instead (or skip the bet)

**Why this is worth testing:** The model's edge exists but is noisy. If the noise
has structure (the model systematically overvalues certain setups), fading those
setups could improve the top-1 strategy further.

**Expected challenge:** This is post-hoc pattern mining, high risk of overfitting.
Any pattern must be identified on training data only (seasons < test season) and
validated out-of-sample. This is more of a research program than a single strategy.

---

### Strategy G2: Multi-Week Rolling Performance Gate

**Hypothesis:** The model runs hot and cold. A strategy that only bets when the
model has been performing well recently (e.g., top-1 disagreement is ≥60% correct
over the last 4 weeks) might avoid cold streaks that kill bankrolls.

**Mechanics:**
- Track a rolling 4-week accuracy on top-1 disagreement bets
- Only place bets when rolling accuracy ≥ threshold (50%, 55%, 60%)
- If below threshold, skip the week
- Compare to always-bet baseline

**Why this could help:** If model performance is autocorrelated (good model weeks
cluster), this gates bad stretches. If it's not autocorrelated, this just reduces
N without improving win rate — which the season-block bootstrap will reveal honestly.

**Implementation notes:**
- This is a simple stateful filter on the walk-forward
- Must use expanding window to avoid look-ahead: week N's gate uses weeks 1…N−1 only

---

## Prioritized Recommendation

Sorted by expected ROI per unit of implementation effort:

| Priority | Strategy | Effort | Rationale |
|---|---|---|---|
| **1** | D1: Disagreement × Confidence Quadrant | Low | Simple post-hoc filter on existing top-1; tests the most obvious interaction |
| **2** | C1: Availability-Driven Bets | Low | Leans into the one proven signal; key driver already computed |
| **3** | B1: Primetime Filter | Low | Just needs kickoff-time tagging; well-known market-structure angle |
| **4** | D2: Confidence-Direction 2×2 | Medium | Extends the proven dogfade finding; higher dof needs careful bootstrap |
| **5** | A1: Fractional Kelly | Medium | Requires bankroll tracking but addresses the fundamental flat-bet limitation |
| **6** | E1: Spread Betting | Medium | Opens a new bet type with lower vig; needs spread-to-prob mapping |
| **7** | B3: Late-Season Filter | Low | Simple temporal split; tests data-quantity hypothesis |
| **8** | C2: Elo-Only Minimal Model | Medium | Requires training a stripped model; tests whether complexity adds noise |
| **9** | E2: Closing Line Value | High | Data dependency (opening lines) but answers the gold-standard question |
| **10** | F2: Post-Blowout Rebound | Low | Simple margin-based filter on existing data |
| **11** | B2: Rest Disparity Threshold | Low | Leverages existing rest_diff feature |
| **12** | G2: Rolling Performance Gate | Low | Stateful filter; tests autocorrelation of model performance |
| **13** | A2: Edge-Proportional Staking | Low | Simple alternative to Kelly |
| **14** | B4: Division Split | Low | Uses existing div_game feature |
| **15** | E3: Totals Correlation | Medium | Exploratory; weak prior |
| **16** | F1: Futures/Preseason Filter | Medium | Needs new data source |
| **17** | G1: Anti-Jinx Pattern Mining | High | Research program, high overfitting risk |

---

## Universal Guardrails

Every strategy above should be evaluated with the same discipline established in §7f:

1. **Season-block bootstrap** (not bet-level) for all CIs and P(ROI>0). The top-2
   and crossover failures are cautionary tales — bet-level bootstraps flatter
   single-season outliers.
2. **Walk-forward only.** No strategy may use information from the test season
   (including threshold optimization). Parameters must be set on training data.
3. **Pre-register thresholds.** The crossover post-hoc min-gap filter is a warning —
   thresholds chosen after seeing results don't count even if the bootstrap doesn't
   penalize them.
4. **Report N honestly.** Small-sample strategies (availability events, primetime-only)
   will have wide CIs. Report them rather than hiding them.
5. **Compare to the top-1 paper play.** The bar for a new strategy isn't "positive ROI"
   — it's "better than the already-tracked top-1 disagreement strategy on the same
   confidence interval."