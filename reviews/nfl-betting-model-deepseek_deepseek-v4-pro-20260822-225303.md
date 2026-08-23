# Technical review: nfl-betting-model

- Model: `deepseek/deepseek-v4-pro`
- Date: 2026-08-22 22:53
- Files reviewed: 64 (~353 KB)

---

**1. Overview**  
This is a mature, well‑engineered NFL win‑probability forecasting system. It ingests play‑by‑play, schedule, snap‑count, and Madden rating data, produces a calibrated logistic or gradient‑boosted model, and surfaces predictions through a CLI, a Streamlit dashboard, and a read‑only cloud app. The codebase is extensively documented (RATING.md, GUIDE.md) and includes multiple validation scripts that isolate feature contributions and guard against leakage.

**2. Strengths**  

- **Thorough leakage control:** The documentation (RATING.md) and code detail how every outcome‑derived feature is shifted so that a game never sees its own result. Rolling windows, expanding windows, and carry‑forward for future games are all correctly implemented.  
- **Rigorous feature vetting:** Each candidate feature (coaching, QB‑EPA, weather) is subjected to a walk‑forward benchmark and, when it proves null, gated behind a flag and kept out of the live path – an exemplary discipline.  
- **Clean separation of concerns:** Data loading, feature assembly, model training, evaluation, and reporting live in distinct modules. The cloud (read‑only) dashboard is a thin renderer that never touches heavy data, making it deployable on a free tier.  
- **Time‑aware calibration:** The training pipeline (model.py) holds out the latest training season to fit the Platt calibrator, avoiding the catastrophic over‑fit that isotonic exhibited in the calibration study.  
- **Production‑ready reporting:** The weekly preview, season tracker, pick’em leaderboard, and paper‑trade tracker are all built on the same validated pipeline with persistent caching.  

**3. Key findings**  

### Critical: Post‑game snap counts used to define “starters” (data leakage)  
- **Files:** `nfl_betting_model/starters.py` (function `starter_unit_ovr`), `nfl_betting_model/features.py` (usage in `build_features`)  
- **Why it matters:** The function loads per‑game snap‑count data (`nfl.load_snap_counts`) and defines “starters” as players whose *offense_pct* or *defense_pct* in that same game exceed 50%. It then averages those players’ Madden ratings to produce features (`ol_ovr_diff`, `dl_ovr_diff`, etc.). This uses post‑game information (who actually played most of the game) to predict the game’s outcome – a clear leakage.  
- **Fix:** Replace per‑game snap percentages with a season‑to‑date rolling average (or prior‑game average) of snap share. The `roster.py` module already computes season‑long averages; adapt that approach so that “starter” for game *i* is determined using snap counts from games *1…i‑1* only. This will make the feature strictly pre‑game.  

### Medium: Starting QB identified via post‑game dropback counts (possible leakage)  
- **Files:** `nfl_betting_model/qb.py` (function `starting_qb`)  
- **Why it matters:** The starter is chosen as the passer with the most dropbacks *in that game’s play‑by‑play*. In the rare case of a last‑minute QB change (e.g., a late scratch), the model retrospectively knows the actual starter – information that was not available before kickoff. While most starters are announced early, this is a subtle but avoidable leak.  
- **Fix:** Use pre‑game sources such as the NFL injury report or depth charts to designate the probable starter. Alternatively, fall back to the most recent week’s starter and only update after the game.  

### Medium: Hard‑coded default Neo4j password  
- **File:** `nfl_betting_model/graph.py` (line `DEFAULT_PASSWORD = "password"`)  
- **Why it matters:** If the `NEO4J_PASSWORD` environment variable is not set, the store connects using `neo4j/password` – a well‑known default that could compromise a publicly accessible database.  
- **Fix:** Raise an explicit error when no password is provided (e.g., `RuntimeError("NEO4J_PASSWORD must be set")`). The default is never safe, even for development.  

### Medium: External weather API not cached or audited  
- **File:** `nfl_betting_model/llm_picker.py` (function `game_weather`)  
- **Why it matters:** Every pick‑generation call hits `wttr.in`, a third‑party service with no SLA, exposing the user’s IP and adding instability. An outage in that service would break the AI expert picker.  
- **Fix:** Fetch weather once per game (or per city+date) and cache the result locally, or use the nflreadpy weather fields (which are post‑game but acceptable for historical training). The cloud app should avoid live API calls entirely.  

### Low: Unverified download of Madden data  
- **File:** `nfl_betting_model/madden.py` (function `_cached_parquet`)  
- **Why it matters:** `urllib.request.urlretrieve` retrieves raw GitHub content without verifying the integrity of the download (no hash check, no TLS pinning). A compromised upstream URL could inject malicious data.  
- **Fix:** Pin a specific commit and compute a content hash after download; abort if the hash mismatches.  

### Low: `detect_target` may return an incomplete week for grading  
- **File:** `nfl_betting_model/weeks.py` (function `detect_target`, mode `"grade"`)  
- **Why it matters:** It returns the maximum week number with any completed game, even if that week is partially played (e.g., a Monday night game not yet final). The grader then scores only completed games, so the effect is benign, but a schedule with a gap could yield an unintended week.  
- **Fix:** Ensure that *all* games of the chosen week are completed before grading, or make the grade mode return the most recent *fully completed* week.  

**4. Suggestions**  

- **Add automated leakage tests:** Write unit tests that construct a small dataset, train a model, and assert that the model’s probability for a game is insensitive to shuffling future data (e.g., scrambling scores after the game).  
- **Centralize environment and API key handling:** The `.env` loader is duplicated in `llm_picker.py`; move it to a shared config module that also validates required keys early.  
- **Refactor `paper.py` what‑if constants:** Allow `WHATIF_REF_EDGE` and `WHATIF_MAX_MULT` to be configured via environment or CLI for experimentation without code changes.  
- **Improve starter feature migration:** Once the leakage is fixed, re‑run the walk‑forward validation to measure the true contribution of the starter‑talent features; the current performance estimates are inflated.  
- **Consider moving the graph ingestion (`ingest_graph.py`) to a separate package or a make‑target:** It adds Neo4j and heavy dependencies but is independent of the betting model; splitting it would lighten the core dependency tree.  

**5. Quick wins**  

- [ ] Remove the default Neo4j password and raise a clear error. (`graph.py:17`)  
- [ ] In `starters.py`, add a large comment warning that the per‑game snap‑count usage leaks and must be fixed before trusting the model’s reported accuracy.  
- [ ] Cache `wttr.in` responses for the season‑week combination in `llm_picker.py` so repeated AI‑picker runs don’t hammer the external service.  
- [ ] In `madden.py`, emit a warning if the downloaded parquet file size differs significantly from the expected size (a cheap integrity check).  
- [ ] Document the QB leakage caveat in `qb.py` docstring so future maintainers are aware.
