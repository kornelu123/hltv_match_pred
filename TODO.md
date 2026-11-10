# HLTV Match Prediction: Plan and TODO

Goal: predict outcomes of upcoming Counter-Strike matches across the markets bookmakers offer, with calibrated probabilities that are evaluated continuously against realized results.

Status legend: `[ ]` todo, `[~]` in progress, `[x]` done, `[?]` blocked on a design decision (see Decisions).

---

## 0. Scope

### 0.1 Market tiers

| Tier | Markets | Data needed | Status |
|---|---|---|---|
| 1. Match | Series winner, map winner (per map), total maps, map handicap, correct score | Results, map history, rosters | Planned |
| 2. Map detail | Total rounds, round handicap, pistol rounds, player kills O/U, player rating/ADR O/U, first kill | Per-map player stats, round or half-score data | Planned, data source [?] D2 |
| 3. Live | Next round, next map, in-play series winner, live player props | Low-latency live feed | Deferred [?] D1 |

### 0.2 Out of scope (until decided otherwise)
- Live/in-play markets (Tier 3) until a live data source is chosen.
- Placing bets, bankroll management, or any claim of guaranteed profit.
- Hosted web app. Output is CLI and files.

### 0.3 Market rules to pin down before modelling
- [ ] Overtime: do OT rounds count toward total rounds and round handicap? (bookmaker convention varies; pick one and document it)
- [ ] Forfeits, walkovers, and abandoned matches: void, or settle by the recorded result? Default: void, and exclude from training and scoring.
- [ ] Map pick/ban: how the map sequence is determined for BO3/BO5 (decided pre-match vs. unknown). Model the map pool as uncertain unless the pick/ban is known.
- [ ] Map handicap line (e.g. ±1.5) and total maps line (e.g. 2.5) conventions for BO3 and BO5.
- [ ] Correct score categories per format (BO3: 2-0, 2-1, 1-2, 0-2; BO5: 3-0 through 0-3).
- [ ] Tie-breaking and ordering of team A/B: fixed by HLTV match ID order, never by result, to avoid label leakage.

---

## 1. Compliance and data source

### 1.1 Facts from HLTV robots.txt (checked)
- Allowed in general: `/matches` (plain), match pages, team and player pages, unfiltered `/results` pages, sitemaps.
- Disallowed: `/matches?*`, `/results?*` with filter params, `/stats?*` with filters, `/events/archive`, `/download/demo*`, `/blog/*`, `/legacystats/*`, `/forums/*`, `/transfers` with filter params, `/search/?*offset*`.
- No `Crawl-delay` listed.
- Sitemaps: `sitemap_index.xml`, `news-sitemap.xml`, `author-sitemap.xml`.

### 1.2 Implications
- The demo download path (`/download/demo*`) is disallowed. Round-level data cannot come from HLTV demo files under this plan. See D2.
- Fetcher must enforce robots rules in code (parser + path check), not by convention.

### 1.3 Compliance tasks
- [ ] Read HLTV terms of service and record the conclusion in this file (section 1.4).
- [ ] Implement robots-aware fetcher: check every URL against Disallow rules before requesting.
- [ ] Conservative default rate limit (configurable; start at one request every few seconds).
- [ ] Persistent raw-HTML cache keyed by URL, storing `fetched_at`, sha256, HTTP status.
- [ ] Never request the disallowed paths, even indirectly via links.
- [ ] Identify the client honestly (User-Agent with contact field); do not spoof a browser to evade blocks. If bot protection blocks plain requests, stop and make a decision (D7).

### 1.4 Terms-of-service conclusion
- [ ] (to fill in after reading ToS)

---

## 2. Architecture

### 2.1 Repository layout

```
hltv_match_pred/
  pyproject.toml
  TODO.md
  README.md
  src/hltvpred/
    config.py            # settings: rate limit, paths, model params
    scrape/
      robots.py          # robots.txt parse + allow check
      fetcher.py         # rate-limited, cached, robots-aware HTTP
      sitemaps.py        # enumerate URLs from sitemaps
    parse/               # HTML -> typed records; no I/O
      upcoming.py        # /matches listing
      match_page.py      # match details, maps, lineups, player stats
      results.py         # results index
      team_page.py
      player_page.py
      rankings.py
    store/
      schema.sql
      db.py              # DuckDB access, Parquet export
    ratings/
      elo.py
      glicko.py          # optional, see D4
    features/
      team.py
      player.py
      roster.py          # roster as of a timestamp
      context.py         # event tier, rest days, LAN/online
      build.py           # assembles feature rows as of cutoff time
    models/
      map_win.py         # map-level round/map win probability
      round_sim.py       # round-level Monte Carlo (Tier 2)
      series.py          # map sequence sampling -> series markets
      props.py           # player prop derivation
      calibration.py
    markets/
      spec.py            # market definitions and rules (section 0.3)
      derive.py          # turns simulated outcomes into market prices
      value.py           # fair prob, edge, edge LCB, threshold methods A/B, flags (section 8.1)
    eval/
      backtest.py        # walk-forward
      metrics.py         # log loss, Brier, calibration, ECE
      report.py
    predict/
      upcoming.py        # predict listed upcoming matches
      log.py             # append-only prediction log
      score.py           # score logged predictions after results
    cli.py               # `hltvpred` entry point
  data/
    raw/                 # gitignored HTML cache
    db/                  # gitignored DuckDB file
    parquet/             # gitignored exports
  tests/
    fixtures/            # saved HTML pages, one per parser
    unit/
    leakage/
  docs/                  # only if needed; keep TODO.md as the plan
```

### 2.2 Stack (default, see D5)
- Python 3.12, managed with uv.
- DuckDB for storage, Parquet for exports.
- pydantic for parsed records, httpx for HTTP, selectolax for HTML.
- pandas or polars for feature tables (pick one; polars preferred).
- scikit-learn for logistic regression and calibration; LightGBM for gradient boosting.
- numpy for Monte Carlo simulation.
- pytest, ruff.

### 2.3 Core design principle: one coherent model
Bookmakers price all markets from one underlying model. This plan does the same (D3 default):

1. **Map win probability and round model**: P(team wins round | map, side, team and player strength, recent form).
2. **Map simulation**: simulate rounds to a map result. Gives map winner, total rounds, round handicap, overtime, pistol-based outcomes.
3. **Series simulation**: sample the map sequence (pool weighted by team map rates, or a pick/ban model). Gives series winner, total maps, map handicap, correct score.
4. **Player props**: per-map kill and rating rates × simulated rounds. Gives kills O/U, rating O/U.
5. **Coherence check**: all markets come from the same sampled outcomes, so no market contradicts another.

Known weakness to address: rounds are not independent (momentum, economy, eco rounds). The round model must be validated against empirical map round distributions, not assumed.

---

## 3. Data model

All tables keyed by HLTV numeric IDs. Every row from scraped pages carries `fetched_at`.

| Table | Key columns | Notes |
|---|---|---|
| `raw_pages` | url, fetched_at, sha256, status, path | Every fetch, cache index |
| `events` | event_id, name, tier, start_date, end_date, location_type, prize_pool | Tier and LAN/online for context features |
| `matches` | match_id, start_time_utc, event_id, format (bo1/bo3/bo5), team_a_id, team_b_id, status, winner_id, is_forfeit, fetched_at | Team A = lower HLTV ID (not result-based) |
| `maps` | match_id, map_order, map_name, team_a_rounds, team_b_rounds, ot_rounds, picked_by | Per-map results |
| `map_half_scores` | match_id, map_order, team_id, half, rounds | Only if available on match pages (D2) |
| `lineups` | match_id, team_id, player_id, as_of | Lineup used for each prediction |
| `player_map_stats` | match_id, map_order, player_id, kills, deaths, assists, adr, rating, opening_kills, opening_deaths | Per-map player stats |
| `team_rank_snapshots` | team_id, snapshot_date, rank, points | Dated rankings only, never current rank for past matches |
| `odds` | source, match_id, market, selection, decimal_odds, observed_at | Benchmark only, never model input (D4 in odds) |
| `predictions` | prediction_id, made_at, match_id, market, selection, prob, fair_odds, model_version, inputs_hash, odds_used, fair_prob_market, se_edge, edge, edge_lcb, threshold, threshold_method, training_cutoff, flagged, flag_reason | Append-only log; flagged and unflagged both stored |
| `prediction_results` | prediction_id, resolved_at, outcome, log_loss_term, brier_term | Filled when results land |

### 3.1 Data tasks
- [ ] `schema.sql` with all tables above, constraints on keys and nullability.
- [ ] Migration/versioning plan for schema changes (simple numbered SQL files).
- [ ] Parquet export of each table for offline analysis.

---

## 4. Phases

### Phase 0: Foundations
- [ ] `pyproject.toml`, uv lock, ruff config, pytest config.
- [ ] `config.py` with rate limit, paths, user agent contact field.
- [ ] Robots parser + allow-check, with unit tests using the real robots.txt saved as a fixture.
- [ ] Fetcher: rate limit, cache, retries with backoff, status logging.
- [ ] Verify plain Python request to `/matches` and one match page. Record result in section 1.3. If blocked, stop and resolve D7 before continuing.

**Exit:** fetcher retrieves allowed pages, refuses disallowed ones, and reruns offline from cache.

### Phase 1: Tracer bullet (match winner only)
- [ ] `parse/upcoming.py`: extract listed upcoming matches (match ID, teams where shown, event, date, format if shown, listed odds). Fixture test from saved `/matches`.
- [ ] `parse/match_page.py`: teams, format, event, start time, lineup status. Fixture test.
- [ ] Seed Elo from a small historical result set (manually fetched, unfiltered results pages).
- [ ] `predict/upcoming.py` + `predict/log.py`: predict series winner for matchups with both teams resolved; skip TBD/qualifier placeholders with reason.
- [ ] `hltvpred upcoming` CLI: table of date, event, teams, format, P(A), P(B), lineup status, skipped reason.
- [ ] Log every prediction.

**Exit:** `hltvpred upcoming` prints predictions for the live listing; predictions are written to the log.

### Phase 2: Historical ingestion
- [ ] `sitemaps.py`: enumerate match and team/player URLs from sitemaps.
- [ ] `parse/results.py`: unfiltered results pages, backfilled from a start date (D6).
- [ ] `parse/match_page.py`: maps, map scores, pick/ban, lineups, per-map player stats, half scores if present.
- [ ] `parse/team_page.py`, `parse/player_page.py`, `parse/rankings.py` (dated snapshots only).
- [ ] Backfill job: resumable, idempotent, respects cache.
- [ ] Parser fixtures for each page type, including a forfeit, an overtime map, a BO5, and a page with missing stats.
- [ ] Data-quality report: counts per table, parse failures, missing fields.

**Exit:** backfill completes for the chosen date range with zero unhandled parse errors; quality report reviewed.

### Phase 3: Ratings and features
- [ ] Elo with map-level updates; BO3/BO5 handling; forfeits excluded.
- [ ] Glicko-2 only if D4 selects it.
- [ ] Team features: Elo, rolling win rate (last 10, 20), map-pool win rates, days since last match, head-to-head (small-sample shrinkage).
- [ ] Roster-as-of features: roster stability, player count of matches together, aggregate player strength.
- [ ] Player features: rolling rating, KPR, ADR, opening duel rate, shrunk toward population mean.
- [ ] Context features: event tier, LAN/online, prize pool (log scale).
- [ ] Antisymmetric difference features; A/B swap augmentation for training.
- [ ] `features/build.py` takes a cutoff time and returns features using only data strictly before that time.
- [ ] **Leakage tests** (required):
  - [ ] No feature for match M reads any row with timestamp ≥ M.start_time.
  - [ ] Upcoming-match features use only rows with timestamp < prediction time.
  - [ ] Rankings used are the latest snapshot ≤ cutoff.
  - [ ] Lineups used are the lineup as of cutoff.

**Exit:** leakage tests pass; feature table reproducible from the cache.

### Phase 4: Tier 1 models and evaluation
- [ ] `models/map_win.py`: map-level win probability. Logistic regression baseline, then LightGBM (D4 in models).
- [ ] `models/series.py`: sample the map sequence; derive series winner, total maps, map handicap, correct score.
- [ ] `markets/derive.py`: implement Tier 1 markets per section 0.3 rules.
- [ ] `models/calibration.py`: Platt or isotonic on out-of-time data.
- [ ] `eval/backtest.py`: walk-forward by month. Never random splits.
- [ ] Baselines: coin flip, rank-only, Elo-only, and market-implied probability (where odds exist, after removing margin).
- [ ] Metrics: log loss, Brier, accuracy, calibration curve, ECE, per format, per tier, LAN vs online.
- [ ] Report written to `eval/report.py` output.
- [ ] Replay check (section 5): held-out past matches, features as of start time, prediction vs recorded result; excluded from training; reported as a smoke test alongside aggregate metrics.

**Exit:** model beats Elo baseline on log loss in most walk-forward folds; calibration plots reviewed; results recorded in the report.

### Phase 5: Round simulator and Tier 2 markets
Depends on D2 (round-level data source).
- [ ] `models/round_sim.py`: round-win model per map and side.
- [ ] Monte Carlo map simulation (numpy, vectorized, configurable sample count).
- [ ] Validate simulated round totals and overtime frequency against empirical distributions from history.
- [ ] Derive total rounds O/U, round handicap, pistol-based markets (only if data supports pistols).
- [ ] `models/props.py`: player kills O/U and rating O/U from per-map rates × simulated rounds; shrink low-sample players.
- [ ] Coherence check: every market derived from the same sample set; assert no contradictions (e.g., correct score sums match series winner).
- [ ] Evaluate Tier 2 markets with calibration and log loss against realized outcomes. Market benchmark only where odds history exists (D3).

**Exit:** Tier 2 markets evaluated in walk-forward; coherence assertions pass on all upcoming matches.

### Phase 6: Outputs and daily loop
- [ ] Output formats per D6 (probabilities, fair decimal odds, or both). Fair odds = 1 / probability, with margin excluded, labelled clearly.
- [ ] Odds snapshots: store every observed price with `observed_at` on each scheduled refresh; record the last price before kickoff as the closing price.
- [ ] Value flags per section 8.1: compute `fair_prob_market`, `se_edge`, `edge_lcb`, and the active threshold; flag; show all in output.
- [ ] `predict/score.py`: resolve logged predictions when results land; write `prediction_results`.
- [ ] Accuracy report: rolling log loss, Brier, and calibration by market since go-live.
- [ ] Run mode per D5. Scheduled refresh interval per D5 (default 60 minutes).
- [ ] Re-predict when a lineup changes; log both predictions, never overwrite.
- [ ] Drift monitoring: alert when rolling calibration error exceeds a threshold.
- [ ] CLV and filter report: closing line value, flagged vs unflagged log loss, realized ROI with CI and sample size (section 8.1).

**Exit:** scheduled job runs unattended for a week; scoring pipeline matches manual checks.

### Phase 7: Tier 3 live markets (conditional)
- [ ] Only start if D1 selects it and a live data source is approved (D8).
- [ ] Re-evaluate the robots/ToS situation for that source before writing any code.

---

## 5. Testing

- **Parser fixtures:** one saved HTML page per parser. Test the fields that matter (IDs, scores, map order, player IDs, odds). Include edge pages: forfeit, overtime, BO5, missing stats.
- **Robots:** test that disallowed paths are refused and allowed paths pass, using the saved real robots.txt.
- **Leakage (most important):** tests in section 4 Phase 3.
- **Market coherence:** tests that derived markets agree with each other from the same sample set.
- **Backtest integrity:** test that splits are chronological and non-overlapping.
- **Replay check:** for held-out past matches, build features as of each match's start time, predict, and compare against the recorded result. Replayed matches must be excluded from training, or the check measures memorization. Report per-match hits and misses as a smoke test only; one match cannot validate a probability. Acceptance is decided by the aggregate log loss and calibration from Phase 4.
- **Metrics:** unit tests against hand-computed log loss and Brier values on tiny inputs.

Not tested: wording or incidental implementation details of the parsers beyond their extracted values.

---

## 6. Evaluation policy

- Primary metric: log loss per market, against the best available baseline.
- Secondary: Brier score, calibration curve, ECE.
- Market benchmark: only where historical odds exist (D3). Without odds, report calibration against realized outcomes only, and say so explicitly.
- Value filter: closing line value and realized ROI with sample size and CI (section 8.1). No "beneficial" claim without that evidence.
- Live accuracy from the prediction log is the authoritative number. Backtests are necessary but not sufficient.
- Report results per market, format, tier, and LAN vs online. Never report a single aggregate accuracy.

---

## 7. Risks

| Risk | Mitigation | Owner |
|---|---|---|
| HLTV blocks plain Python requests | Test in Phase 0; stop and decide D7 before proceeding | Phase 0 |
| Disallowed demo path needed for round data | Use D2; never fetch `/download/demo*` | Phase 5 |
| Lookahead leakage | Leakage tests; cutoff-based feature builder only | Phase 3 |
| Sparse player data and roster churn | Shrinkage; roster-as-of; roster-stability feature | Phase 3 |
| Round independence assumption is wrong | Validate simulated distributions; calibrate | Phase 5 |
| Market prices we can't benchmark (no odds history) | Calibration-only evaluation; say so in reports | Phase 4 |
| Overconfidence leading to bad betting decisions | Calibration reporting, never profit claims; output includes uncertainty | Phase 6 |
| Betting value flags look good on small samples, or a learned threshold overfits | Pre-registered method; walk-forward for B; fall back to A; CLV and ROI with CI; no profit claims | Phase 6 |
| No odds floor: very short prices flagged on a thin edge | Uncertainty term scales with odds; review flagged-odds distribution in CLV report; add a floor only by logged decision | Phase 6 |
| HTML markup changes | Fixture tests fail loudly; parse errors counted in quality report | Phase 2 |
| Model claims beat market without evidence | Require walk-forward and live-log evidence before any claim | Phase 4, 6 |

---

## 8. Decisions

Each decision has an ID used in the rest of this file. Status is updated when the user answers.

| ID | Decision | Options | Default | Status |
|---|---|---|---|---|
| D1 | Market scope for first build | Tiers 1+2 now, Tier 1 only, all tiers | Tiers 1+2 now, Tier 3 deferred | [x] Tiers 1+2 now; Tier 3 deferred |
| D2 | Round-level / Tier 2 data source | HLTV match pages only; external source; demos from non-HLTV source | HLTV match pages only (round markets approximated) | [x] HLTV match pages only |
| D3 | Historical odds for benchmarking | None (calibration only); user provides odds files; collect odds going forward | Collect going forward from listing pages | [x] Collect going forward from listing pages |
| D3a | Model granularity | Round-simulator core; independent per-market models | Round-simulator core | [x] Round-simulator core |
| D5 | Run mode and refresh | Manual CLI; scheduled job (refresh interval, minutes) | Scheduled, 60 minutes | [x] Scheduled, 60 minutes |
| D6 | Output format | Probabilities; fair decimal odds; both | Both | [x] Both |
| D4 | Rating system | Elo only; Elo + Glicko-2 | Elo only | [ ] |
| D5b | Storage | DuckDB + Parquet; SQLite; Postgres | DuckDB + Parquet | [ ] |
| D7 | If plain requests are blocked | Stop and ask; browser-driven fetch (Playwright); other | Stop and ask | [ ] |
| D8 | Live data source for Tier 3 | None; name a source (checked for robots/ToS first) | None | [ ] |
| D9 | Historical start date for backfill | e.g. one year back; longer | One year back | [ ] |
| D10 | Value filter threshold (section 8.1) | A: uncertainty-adaptive; B: learned from logged CLV; C: A now, B later | C, z = 1.0, no odds floor | [x] Chosen: C, z = 1.0, no odds floor |

Notes:
- D3a and D3 both affect how odds are used; odds are never a model input by default.
- D7 and D8 are blockers only if reached.

### 8.1 Value filter: dynamic threshold (approach C)

Pre-registered: the method and its parameters, not fixed threshold values. The model computes each threshold per prediction from data available at prediction time.

Flag rule, evaluated per prediction against the prices seen then:

- `edge = p_model × decimal_odds − 1` (expected return per unit stake under the model)
- `se_edge = se_p × decimal_odds`, where `se_p` is the spread of `p_model` across ensemble members (Tier 2: Monte Carlo sampling error plus parameter spread)
- `edge_lcb = edge − z × se_edge`, with `z = 1.0` (D10c)
- flag if `edge_lcb ≥ threshold`, where `threshold` comes from the active method below
- both sides of the market are priced, and the market is not drift-suppressed (Phase 6)

No static odds floor (D10b). The uncertainty term is the only protection against short prices; see the risk table.

Threshold methods:
- **A, uncertainty-adaptive (always available):** `threshold = 0`. The lower bound carries the uncertainty, so a bet is flagged only when the model is confident the edge is positive. Needs no odds history.
- **B, learned (active once enough labelled bets exist):** a model predicts expected closing-line value (CLV) for a candidate bet from `p_model`, `decimal_odds`, `se_p`, market type, format, event tier, LAN/online, lineup stability, and days since last roster change. Flag when predicted CLV ≥ 0 and `edge_lcb ≥ 0`. Trained only on resolved bets before the training cutoff; retrained on a schedule (weekly).
- **Switch A → B:** only when B's walk-forward CLV is positive with a confidence interval that excludes zero, and the labelled-bet count reaches a minimum set at adoption time (not now).

Overfitting protection (required for B):
- Training data ends before each prediction. Never retrain on the bets being judged.
- Evaluate on forward bets only (walk-forward). Report held-out CLV and ROI, never training fit.
- Fall back to A if B's walk-forward evidence is not positive.
- Store `threshold_method`, `threshold`, model version, and `training_cutoff` with each prediction.
- Report threshold stability: how far the learned threshold moves between retrains.

Labels: CLV needs closing-price snapshots, which exist only from go-live (Phase 6 odds snapshots). Until then, B has no training labels.

Logging and evaluation: `predictions` stores `odds_used`, `fair_prob_market`, `se_edge`, `edge`, `edge_lcb`, `threshold`, `threshold_method`, `flagged`, `flag_reason`. Flagged and unflagged predictions are both logged. Evaluate as in section 6: CLV, realized ROI with sample size and bootstrap CI, flagged vs unflagged log loss.

Scope limits:
- The listing shows odds for match winner only. Flags apply only where an observed price exists.
- No stake sizing.
- Prices move and can disappear. A flag records the price seen, not a price that will be available.

Threshold change log:
- Replaced fixed `min_odds 1.50` / `min_edge 0.05` with the method above (user request).
- Approach C chosen; no odds floor (D10b); z = 1.0 (D10c).

---

## 9. Open questions (to resolve before Phase 1 starts)

- [x] D1, D2, D3, D3a, D5, D6 answered (interactive form); recorded in section 8.
- [ ] D4, D5b, D9 accepted at default or changed (still open; defaults in use).
- [ ] Replay check definition (section 5) confirmed.
- [ ] ToS reviewed; section 1.4 filled in.
- [x] D10 chosen: approach C (uncertainty-adaptive now, learned later), no odds floor, z = 1.0 (section 8.1).
