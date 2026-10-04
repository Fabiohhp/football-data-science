# CLAUDE.md

**This is the project's design brief** — the decisions, conventions and data
traps that everything in this repository follows. It doubles as the working
context for Claude Code, which is why it is written as instructions; if you are
a human reader, §3 (why each tool was chosen), §5 (the leakage traps — the ways a
model here gets a score it did not earn) and §7 (evaluation conventions) are the
parts worth reading. The README is the front door; this is the reasoning behind it.

Read it fully before generating or modifying code. It inherits everything in
`../football-data-analytics/CLAUDE.md` §5 — read that too.

---

## 1. Project Overview

This project is **Part 3 — Machine Learning**, the last part of a football
(soccer) data portfolio. It trains models that **predict what has not happened
yet** — match outcomes, and the chance that a shot becomes a goal — and evaluates
them honestly against baselines.

It is its own repository, sitting next to the two that feed it. **Only the
sibling relationship matters** — every path is relative, so the three can live
anywhere as long as they stay side by side:

```
portfolio/football-data/
├── football-data-engineering/   Part 1 — the pipeline; owns the Gold layer
├── football-data-analytics/     Part 2 — metrics layer, analyses, dashboard
└── football-data-science/       Part 3 — this repository
```

| Phase | Scope | Where | Status |
|-------|-------|-------|--------|
| Part 1 — Data Engineering & Architecture | Ingestion, Airflow, Bronze/Silver/Gold, data quality | `../football-data-engineering` | Complete |
| Part 2 — Analytics & BI | Metrics layer, analyses, dashboard | `../football-data-analytics` | Complete |
| **Part 3 — Machine Learning** | Match outcome and xG models, evaluation | **this repository** | **Phases 0–4 of 6 done; phase 5 built, awaiting its first prospective score** |

**The deliverable is a model a reviewer can trust the score of** — not the
highest score. The audience is someone hiring for data science or ML: they will
read the README, open the evaluation notebook and check the split. Optimize for:

- a baseline for every model, and a comparison against it with an interval;
- evaluation that could not have seen the future — enforced by tests, not care;
- reproducibility from a pinned snapshot, **without running Part 1**.

Part 2 drew the line this repository starts from (its §7): *explaining* what
happened is Part 2; anything trained to *predict* unseen matches — splits,
feature pipelines, model selection, xG models built from `fact_shots` — is here.

---

## 2. Status, and what this repository inherits

**Models #1–#3 and their baselines are built (phases 0–4, §12); #4 is not.** The data is: Part 1's Gold holds three big-5 seasons
(2024/25 and 2025/26 complete, 2026/27 in progress — about 3,700 matches and
94,000 shots as of 2026-09), and Part 2's metrics layer builds on it with 131
passing dbt tests. Treat those counts as a reading; the authority is the
snapshot's `manifest.json`.

**This is a small dataset for ML, and the design follows from that.** Two full
seasons are ~3,500 matches — a few hundred per league per season. That rules out
anything with many parameters, makes the choice of split matter more than the
choice of model, and means differences between models are often inside their own
noise. Every comparison carries an interval (§7).

**History starts in August 2024.** There is no earlier data in Gold, so the first
matchdays of 2024/25 have no form history for anyone, and every club promoted
into a later season arrives with none. That is a cold-start problem to handle
explicitly (§5, trap L7), not a gap to fill.

**This repository never calls Football-Data or Understat, and never writes to
Part 1 or Part 2.** If data is missing, the fix belongs in Part 1; if a metric is
missing, it belongs in Part 2's marts.

---

## 3. Tech Stack

| Concern | Choice | Why |
|---------|--------|-----|
| Environment | **uv**, Python 3.12 | Same toolchain as Parts 1 and 2 |
| Input | **Part 2's `analytics.duckdb`**, attached `READ_ONLY` | The marts are already tested; re-deriving team matches or shots here would duplicate §5 of Part 2 and drift from it |
| Features | **dbt-core + dbt-duckdb**, `feat_*` models | Point-in-time features are where leakage lives; dbt gives each one a stated grain, tests and lineage, and a singular test can *prove* no feature reads the future |
| Models | **scikit-learn**, **statsmodels**, **scipy** | Logistic regression, gradient boosting (`HistGradientBoostingClassifier`, no extra dependency), Poisson GLMs, and a Dixon-Coles likelihood written against `scipy.optimize`. Enough for this volume |
| Evaluation | own `evaluate.py`: RPS, log loss, Brier, calibration, paired bootstrap | Ranked probability score is the standard for ordered H/D/A outcomes and is not in scikit-learn; a dozen lines, tested |
| Analysis | **Jupyter**, matplotlib, mplsoccer | Same as Part 2 — static images render on GitHub |
| Artifacts | `joblib` files + a JSON **model card** per model | See below |
| Quality | ruff, black, pytest, dbt tests, GitHub Actions | Same bar as Parts 1 and 2 |

Considered and not chosen:

- **MLflow / Weights & Biases.** An experiment tracker earns its keep over hundreds
  of runs by a team. Here there are a handful of models, each defined in code; a
  committed model card (training range, snapshot citation, features, scores,
  git commit) records everything a tracker would, and a reviewer can read it on
  GitHub without starting a server.
- **XGBoost / LightGBM.** scikit-learn's histogram gradient boosting is the same
  family, needs no compiled extra, and at a few thousand rows the difference is
  not measurable.
- **Deep learning.** Nothing here has the volume, and a network on 3,500 matches
  would be a worse model and a worse portfolio piece.
- **Reading Gold directly.** Part 2 already solved the traps in Gold and tests
  them. Reading its marts means inheriting those fixes rather than rediscovering
  them.
- **A feature store.** The features are a few dbt tables.

---

## 4. Architecture

```
  ../football-data-analytics/data/analytics.duckdb   (Part 2's marts + int_ models)
          │  ATTACH ... (READ_ONLY) — the manifest travels with it
          ▼
  data/features.duckdb
     src_*     thin selects over the Part 2 models this repository reads
     feat_*    point-in-time features: one row per thing to predict,
               built only from what was known before it happened
          │
          ├──► src/football_ml/     train.py, evaluate.py, models/
          │         │
          │         ▼
          │    models/<name>/<version>/   model.joblib + model_card.json
          │         │
          │         ▼
          │    reports/<name>.json        scores, intervals, calibration bins
          │
          ├──► notebooks/   one question each, reading reports and features
          │
          └──► scripts/export_predictions.py → exports/*.json
                    for Part 2's dashboard to show (§9)
```

### Layer rules

- **Input.** Part 2's database is opened `READ_ONLY`, and the feature build
  refuses a database whose `stg_matches` row count disagrees with its snapshot's
  `manifest.json` — the same staleness check Part 2's `marts.py` makes, for the
  same reason. The manifest's export time and `match_date` range are copied into
  every model card.
- **Contract.** `contracts/part2_columns.json` lists every Part 2 column this
  repository reads, with its type. The build fails on any difference. Part 2 may
  rename a column; this repository must find out at build time.
- **Sources (`src_`).** One model per Part 2 table read. Select and rename only.
  Which Part 2 tables may be read is itself a rule — see trap L2.
- **Features (`feat_`).** Where leakage is prevented. Each states its grain and
  its **as-of column** (the moment the row predicts from) in `schema.yml`, and
  every input row it uses must be strictly before that moment — tested (§6).
- **Training and evaluation** live in `src/football_ml/`, never only in a notebook.
  A notebook calls `train()` and reads the report; it does not fit a model the
  pipeline cannot reproduce.
- **Exports.** Predictions only, for display. No training happens there, and
  nothing is recomputed.

---

## 5. Traps

### Inherited from Part 2

Every trap in `../football-data-analytics/CLAUDE.md` §5 applies. Reading Part 2's
marts instead of Gold already handles most of them — `int_team_matches` applies
the counting filters (traps 6 and 13), `mart_shots` deduplicates and attaches
season, names are unescaped (trap 15) — but four still bite a model directly:

- **Null xG is not zero** (trap 7). Twelve counting matches have no xG. A feature
  built from them must be null, and the model must see "missing", not `0.0`.
- **Own goals are rows in `fact_shots`** (trap 11), with 0.0000 xG, sitting at the
  scorer's own end. They are not shots. Exclude them from any shot model.
- **Penalties sit at one point and carry 0.756 xG** (trap 10). A model trained on
  location will learn them perfectly and flatter its own score. Model non-penalty
  shots, and treat penalties as a constant reported separately.
- **Duplicate shots in two La Liga matches** (trap 14). Use the deduplicated rows.

### The leakage traps

**Each of these produces a model that scores well in evaluation and badly on the
next match**, which is the worst failure available here, because it looks like
success.

**L1. `mart_team_form` includes the match itself.** Its window is
`rows between N-1 preceding and current row`: the rolling xG on a match's row
contains that match's xG. Built for a chart, it is correct there; as a feature it
hands the model the answer. Features shift the window to end *before* the match —
`feat_` models compute their own rolling values from `int_team_matches` rather
than reuse the chart's.

**L2. Season-level marts are summaries of the whole season.** `mart_team_season`,
`mart_team_season_half`, `mart_player_season` and every percentile and band on
them (`points_minus_xpts_rank`, `gap_band`, `npxg_per_90_percentile`) are computed
over matches that had not been played at the time of a mid-season prediction.
They may only be joined as *previous* season information — `season = prior
season`, never the current one. `src_` models over them carry the rule in their
name (`src_team_season_prior`).

**L3. A match's own xG, xPTS, PPDA and deep are outcomes, not inputs.** They are
measured during the match being predicted. Only earlier matches' values may be
features.

**L4. Same-day matches.** `match_date` is a `DATE`, with no kickoff time. A club
cannot play twice in a day, but "strictly before the match" must mean
**`match_date < as_of_date`**, never `<=` — a `<=` on league-wide features (e.g.
the league's scoring rate) would leak the evening's results into the afternoon's
prediction.

**L5. Random splits shuffle time.** K-fold on matches trains on May to predict
the previous August. Every split is **forward in time** (§7). For shot models,
shots within a match are also correlated — split by match, never by shot.

**L6. Tuning on the test set.** A hyperparameter chosen by looking at test scores
turns the test set into training data. The final hold-out is scored **once**, by
a model whose settings were fixed on the validation folds, and the report records
that it was.

**L7. Cold start is not zero.** A club with no earlier matches in the loaded range
— every club in August 2024, every promoted club after — has null form features.
Zero would read as "scored nothing, conceded nothing". Handle it explicitly: a
prior (the league's promoted-club average from earlier seasons, where that
exists), plus a `matches_of_history` feature, and report scores separately for
matches where history was short.

**L8. A player's club is the one in the event row** (Part 2 trap 2), and a player's
history starts at the loaded range (Part 1's SCD2 note). `dim_players.current_team_key`
is today's club, which is the future relative to any past match.

**L9. Understat's xG is a model, and so is the target of an xG comparison.**
Using Understat's `xg` as a *feature* of a shot model is circular; it is the
**benchmark**. The target is `is_goal`.

**L10. Gold does not carry everything Understat knows about a shot.** Understat
sends `lastAction` (the action before the shot — a cross, a through ball, a
rebound), and Part 1 parses it into Silver as `last_action`, but it is **not in
Gold's `fact_shots`**. A shot model here sees location, situation, body part and
assist presence only, and should expect to fall short of Understat's. Say so in
the report rather than explaining the gap away. Adding it to Gold is a Part 1
task — raise it there.

**L11. Draws are hard, and a model can hide it.** A model that never predicts a
draw as the most likely outcome can still post a respectable log loss. Report
calibration **per class**, and the predicted draw rate against the actual one.

---

## 6. Feature layer conventions

- **Grain and as-of are stated and tested.** `feat_match_outcome` is one row per
  counting match, with `as_of_date = match_date`; `feat_shots` is one row per
  non-penalty, non-own-goal shot. Composite grains use Part 2's vendored
  `unique_combination` test, copied, not imported.
- **Every feature row lists the latest input date it used**, as
  `max_input_date`, and **`assert_no_feature_reads_the_future`** fails on any row
  where `max_input_date >= as_of_date`. This is the one test the whole
  repository rests on. It is a **generic test attached to each `feat_` model** —
  not one singular test over all of them, which `dbt build` skips whenever another
  test upstream fails first (§12, phase 1) — and
  `assert_every_feature_model_is_leakage_tested` fails any `feat_` model without it.
  `tests/test_leakage.py` breaks the features on purpose on every run (L1 and L2)
  and asserts this test, by name, is what fails.
- **Rolling windows never cross a season boundary silently.** Either the window
  resets each season (as Part 2's form chart does) or carries over and says so in
  a column; the choice is per feature, stated in `schema.yml`.
- **Coverage travels with the feature.** Every rolling xG feature carries its own
  `*_matches` count, as in Part 2 — a window over four matches with xG and one
  without is not the same number as a window over five.
- **Windows and thresholds are dbt vars**, never literals (`form_window`,
  `min_history`).
- **Features describe the two sides symmetrically.** A match row carries
  `home_*` and `away_*` for every feature, built by one macro, so the model cannot
  see a feature for one side that it lacks for the other.
- **Everything is materialized as a table**, for the reason Part 2 gives: the
  database stays self-contained regardless of the working directory.

---

## 7. Modelling and evaluation conventions

- **Every model has a baseline, and the baseline is in the same report.**
  - Match outcome: the league's historical H/D/A rates (prior seasons only), then
    an Elo rating. A model that does not beat Elo is reported as not beating it.
  - xG: a logistic regression on distance and angle alone — and Understat's own
    xG as the ceiling (L9, L10).
- **Splits are walk-forward.** Train on everything before a date, predict the
  next block (a matchday round or a month), roll forward, repeat. The final
  hold-out is the most recent complete season and is scored once (L6). The
  in-progress season is the live test (below).
- **Primary metrics.** Match outcome: **ranked probability score** (ordered
  H/D/A), with log loss and a per-class calibration plot beside it. Shots: log
  loss and Brier, with a calibration plot. Accuracy is never the headline — it
  ignores the probabilities, which are the product.
- **Compare with intervals.** Model A against model B is a **paired bootstrap
  over matches** of the score difference, seeded (Part 2's `stats.DEFAULT_SEED`
  convention). "B is better" needs the interval to exclude zero; otherwise the
  report says the two cannot be distinguished — as Part 2's question 1 did.
- **Calibration before discrimination.** A probability that says 60% should come
  true about 60% of the time; check that before any ranking metric.
- **Prospective evaluation, because this dataset keeps growing.** Part 1 loads
  2026/27 daily. A model is frozen at snapshot S₁ (card + commit hash), and when a
  later snapshot S₂ arrives, every match with `match_date` after S₁'s range is
  scored — with features rebuilt from S₂, which contain only pre-match data by L4.
  This is the only test in the repository that no amount of care in the split
  could have fooled, and the README reports it separately. Gold holds no future
  fixtures, so there is no forecasting ahead of kickoff; this scores the frozen
  model on matches it never saw.
- **No causal language**, as in Part 2. A feature's importance says what the
  model used, not what wins matches.
- **Do not claim to beat the market.** Gold has no betting odds, so there is no
  bookmaker benchmark here. A good RPS is good against the baselines shown, and
  the README says that is the comparison.
- **Every notebook** follows Part 2 §7: one question, stated first, answered last
  with caveats; run top to bottom on a fresh kernel; outputs committed; a single
  `print` per cell so re-runs reproduce the file.

---

## 8. Model backlog

Ranked. #1 continues Part 2's flagship question: if xG difference carries
information points do not, a model should be able to use it.

| # | Question | Target | Baselines | Needs |
|---|----------|--------|-----------|-------|
| 1 | Can pre-match form — goals, xG, xPTS — predict H/D/A better than league base rates and Elo? | H/D/A from the scoreline of a counting match, as Part 2 derives points | base rates, Elo | `feat_match_outcome` |
| 2 | Does modelling **goals** (Dixon-Coles / Poisson on each side's scoring rate) beat modelling the **result** directly? And does fitting it to xG instead of goals help? | goals per side → H/D/A | #1's best | `feat_match_outcome` |
| 3 | How close can a location-and-context xG model get to Understat's, with what Gold carries? | `is_goal`, non-penalty shots | distance+angle logistic; Understat xG as ceiling | `feat_shots` |
| 4 | Does a player's non-penalty xG per 90 predict their next season better when shrunk towards the league mean? | next season's npxG/90 | last season's raw value | `mart_player_season` (prior season only) |

Deliberately not on the list:

- **Season simulation** (final table probabilities). It needs the remaining
  fixture list, and Gold holds only played matches. A backtest-only version —
  cut at a mid-season date, simulate the known remainder — is possible after #2,
  and must be labelled as a backtest.
- **Player valuation, injury, transfer models.** Gold carries none of that data.
- **Possession-based features.** Not in Gold (Part 2 trap 9).

---

## 9. Showing predictions

Part 2's dashboard already promises this (its §9): **"Part 3 shows predictions
the same way — as pre-computed JSON the site reads, never a model running live."**

- `scripts/export_predictions.py` writes `exports/*.json`: per match, the
  predicted H/D/A probabilities, the actual result, and the model version, plus
  the model card's training range and snapshot citation. Rows ready to display,
  rounded — the same rule as Part 2's export: select and format, never compute.
- The calibration table and the headline scores come from `reports/`, not from
  the frontend.
- **How the files reach the site is an open decision (§13)** — a fifth page in
  Part 2's `site/` reading a copied `exports/` directory is the natural answer,
  and keeps one dashboard. That page then follows every Part 2 §9 rule: no metric
  in JavaScript, answer before table, degrade rather than blank, two engines in CI.

---

## 10. Repository Structure

```
football-data-science/
├── .github/workflows/ci.yml   # lint, pytest, dbt build on the fixture, execute notebooks
├── contracts/
│   └── part2_columns.json     # every Part 2 column read, with types; fails on drift
├── dbt/
│   ├── dbt_project.yml        # form_window, min_history vars
│   ├── profiles.yml           # DuckDB path + READ_ONLY attach of Part 2's database
│   ├── macros/                # both-sides feature macro, unique_combination (copied)
│   ├── models/
│   │   ├── sources/           # src_* — select/rename over Part 2's models
│   │   └── features/          # feat_match_outcome, feat_shots, feat_player_next_season
│   └── tests/                 # leakage-test coverage, recomputation, grain tests
│                              #   (the leakage test itself is a generic in macros/)
├── src/football_ml/
│   ├── data.py                # opens features.duckdb; staleness check; .citation()
│   ├── splits.py              # walk-forward splits — the only place a split is made
│   ├── baselines.py           # base rates, Elo
│   ├── models/                # one module per model family
│   ├── evaluate.py            # RPS, log loss, Brier, calibration, paired bootstrap
│   ├── train.py               # fit → model card → report
│   └── viz.py                 # calibration and reliability plots, Part 2's style
├── models/                    # gitignored except model cards: <name>/<version>/
├── reports/                   # committed: scores, intervals, calibration bins (JSON)
├── notebooks/                 # NN_question.ipynb, outputs committed
├── scripts/
│   └── export_predictions.py  # models + reports -> exports/*.json
├── exports/                   # gitignored until Part 2 §13 decides on publishing
├── tests/                     # pytest: metrics against hand-computed values, splits
│                              #   never overlap in time, Elo updates, card contents
├── data/                      # gitignored except .gitkeep: features.duckdb
├── pyproject.toml
├── uv.lock
├── .python-version            # 3.12
└── README.md
```

**CI and the fixture — two jobs.** CI never has real data, and the repositories are
private, so a workflow here cannot read Part 2 without a credential.

- **`build`** is self-contained and always runs. `tests/synthetic_part2.py` generates a
  stand-in Part 2 from `contracts/part2_columns.json` — every contracted table with
  exactly its columns and types — holding a four-club league over three seasons: a
  season boundary, a relegated and a promoted club, a match with no xG, and a partial
  live season. Lint, pytest (which uses the same generator) and `dbt build` run on it.
- **`part2-integration`** checks Part 2 out beside this repository, builds Part 2's marts
  on Part 2's own fixture — which carries the Gold-level traps the synthetic Part 2 does
  not — and runs `dbt build` on the result. It is how a Part 2 change that breaks this
  repository shows up. It needs the `PART2_DEPLOY_KEY` secret (a read-only deploy key on
  Part 2, README "CI"); without it the job reports it was skipped and passes.

The first design had only the second job. It failed on GitHub because the default token
cannot check out another private repository — the reason `build` no longer depends on it.

---

## 11. Common Commands

> The notebooks are **not built yet**. If a command fails because it is not built, say
> so instead of inventing output.
>
> Paths: `PART2_DIR` (default `../football-data-analytics`) locates Part 2;
> `FEATURES_DUCKDB` (default `data/features.duckdb`) is this repository's database;
> `GOLD_SNAPSHOT_DIR` is **Part 2's** variable with Part 2's meaning — relative to
> Part 2's root — and aims both repositories at the same snapshot. Export
> `GOLD_SNAPSHOT_DIR=tests/fixtures/gold` to work on Part 2's fixture, as CI does.

```bash
uv sync --extra dev

uv run pytest -q
uv run ruff check .
uv run black --check .

# Part 2's marts must be built first — this repository reads them.
(cd ../football-data-analytics && uv run dbt build --project-dir dbt --profiles-dir dbt)

uv run dbt build --project-dir dbt --profiles-dir dbt     # features + leakage tests
uv run python -m football_ml.train match_outcome --validation-only  # develop here:
                                                          #   the hold-out is untouched
uv run python -m football_ml.train match_outcome          # fit, card, report — scores the
                                                          #   hold-out; the ledger counts it
uv run python -m football_ml.train shot_xg                # the shot model, same discipline

# Prospective evaluation (§7). Freeze from a CLEAN tree — it refuses otherwise — and
# commit the card it writes. Then, once Part 2 holds a newer snapshot and the features
# are rebuilt from it, score the frozen model on the matches played since.
uv run python -m football_ml.freeze                       # models/frozen/<id>/freeze_card.json
uv run python -m football_ml.prospective                  # reports/prospective_<id>.json

uv run python scripts/export_predictions.py               # -> exports/predictions/*.json;
                                                          #   Part 2's export copies them
```

---

## 12. Roadmap

Each phase is small and ends in something showable. Finish one before starting
the next.

| Phase | Deliverable | Done when |
|-------|-------------|-----------|
| 0 ◐ | Repo skeleton, READ_ONLY attach of Part 2, column contract, CI on Part 2's fixture | CI builds Part 2 on the fixture and this repository's empty dbt project on top of it — **`build` is now self-contained (§10); the Part 2 job waits on the deploy key** |
| 1 ✅ | `feat_match_outcome`, walk-forward splits, `assert_no_feature_reads_the_future` | The test passes, and fails when a window's shift is removed on purpose |
| 2 ✅ | Baselines + model #1, `evaluate.py`, first report | RPS for base rates, Elo and the model, with paired intervals |
| 3 ✅ | Model #2 (Dixon-Coles on goals and on xG) | Compared with #1 on the same splits |
| 4 ✅ | Model #3 (xG) against Understat's | Calibration plot and log loss beside Understat's, with the L10 gap stated |
| 5 ◐ | Prospective scoring on the next snapshot, predictions page in Part 2's dashboard | A frozen model scored on matches played after it was frozen — **built and tested on synthetic snapshots; the real freeze and its first score are pending** |
| 6 | README as the front page — and the portfolio's closing chapter | A reviewer understands what the models can and cannot do without running anything |

**What phase 0 established.** Part 2's `analytics.duckdb` is attached `READ_ONLY` as
`part2` in `dbt/profiles.yml`, and two `on-run-start` hooks run before any model is
built: `assert_part2_matches_contract` (every column in `contracts/part2_columns.json`
present with its type — extra Part 2 columns are ignored, and nullability is not
compared because dbt-built DuckDB tables report every column nullable) and
`assert_part2_is_current` (Part 2's `stg_matches` count equals its manifest's
`fact_matches`). Both were seen to fail: `tests/test_part2_guards.py` builds a synthetic
Part 2, breaks it one way per test — stale, column dropped, column retyped, table
missing — and asserts the real `dbt build` stops with a message naming the problem;
the passing case also asserts Part 2's file is byte-for-byte unchanged.

Two source models exist: `src_team_matches` (15 columns of `int_team_matches`) and
`src_snapshot`, one row copied from the manifest, so `features.duckdb` carries its own
provenance and later model cards cite it without needing Part 2's checkout. On the real
snapshot: 1,752 counting matches in each complete season, 216 so far in 2026/27, and
2 + 9 + 1 matches with null xG — Part 2's trap 7 exactly.

One bug found by the tests, worth remembering: **`read_json` infers an ISO timestamp
with an offset as a naive `TIMESTAMP` and drops the offset**, so casting it to
`timestamptz` afterwards reads it as local time — `exported_at` came out three hours
wrong in UTC-3 and would have been right in CI, which runs in UTC. `src_snapshot`
extracts the raw string with `->>` and casts that. Read timestamps from JSON that way.

**What phase 1 established.** `feat_team_form` (team × match) and `feat_match_outcome`
(one row per counting match, 17 features per side as `home_`/`away_` pairs from one
macro) are built from `src_team_matches` and `src_team_season_prior` — Part 2's
`mart_team_season` re-keyed by `applies_to_season`, so the only easy join is the L2-safe
one. Form carries over seasons (with `form_prior_season_matches`); season-to-date resets;
nothing is imputed. On the real snapshot: the 48 matches of 2024/25's opening round have
no input at all; 85% of 2025/26 home sides have a prior season (the rest were promoted);
the smallest gap between a feature's latest input and its match is 3 days.

The leakage test was broken on purpose, and **the first version failed to fail.** With the
form window ending at `current row` (L1), `dbt build` failed — but on
`assert_form_matches_a_direct_recomputation`, and the leakage test was never run: it was
a singular test reading both `feat_` models, and `dbt build` skips everything downstream
of a failing test, so `feat_match_outcome` was never built. A leak elsewhere, with no
second test beside it, could have passed a build that skipped the one test meant to catch
it. It is now a generic test on each `feat_` model, and `tests/test_leakage.py` asserts
**`FAIL … assert_no_feature_reads_the_future_feat_team_form`** for both leaks (L1: 4
rows; L2, the season summary joined on the current season: 4 rows), plus that a new
`feat_` model without the test fails the build. Those run on every `pytest`.

Splits (`splits.py`): validation is monthly walk-forward from November 2024 (three months
of burn-in) to May 2025 — 7 folds, 1,300 matches; the hold-out is 2025/26, the latest
season in which every league played `clubs × (clubs − 1)` matches — 10 folds, 1,752
matches; 2026/27 is in no fold.

**What phase 2 established.** Model #1 does **not** beat Elo. On the 2025/26 hold-out:

| | RPS | log loss | Brier |
|---|---|---|---|
| league base rates | 0.2293 | 1.0731 | 0.6489 |
| Elo (k = 30, ordered logit) | **0.2076** | **1.0103** | **0.6046** |
| model #1, `logistic(C=0.03)` | 0.2091 | 1.0225 | 0.6105 |

Model − Elo, RPS: **+0.0015 [−0.0030, +0.0058]** — cannot be distinguished, and the point
estimate favours Elo. Both beat base rates clearly (model −0.0202 [−0.0272, −0.0130],
Elo −0.0217 [−0.0288, −0.0146]). This is Part 2's question 1 again, from the predictive
side: form, xG and xPTS over the last five matches add nothing measurable to a rating
that has absorbed every result since August 2024.

Settings were chosen on validation only (`--validation-only` runs that and stops), then
the hold-out was scored once; `reports/match_outcome.json` carries a ledger of hold-out
evaluations keyed by a fingerprint of the settings, features and folds. A re-run with
nothing changed reproduced the report exactly and left the ledger at 1.

Worth knowing before building on it:

- **Both Elo and the model are overconfident at the extremes.** Where they give an away
  win 60–70%, it happens 52% of the time; the favourite wins less often in 2025/26 than
  either promised. A calibration step (or a less extreme Elo `k`) is the obvious lever,
  and must be chosen on validation.
- **Draws (L11):** actual rate 25.5%; base rates and Elo never make a draw the most likely
  outcome, and the model does in 6.6% of matches — but where it says 40–50% draw, 26%
  are draws. Its draw picks are wrong.
- **Short history (L7):** in the 65 hold-out matches where a side had fewer than
  `min_history` earlier matches (promoted clubs, early season), the model scores 0.197
  against Elo's 0.214. Sixty-five matches and no interval: a direction, not a finding.
- **Histogram boosting lost to the logistic regression** on validation, getting worse with
  every step of depth or iterations — ~1,300–1,750 training matches are too few for it.
- Two practical traps: DuckDB's nullable integers reach pandas as `Int64` with `pd.NA`,
  which scikit-learn refuses (the model casts to `float64`, so missing stays `NaN`); and
  `prior_season_*` is empty for every row of 2024/25, which histogram boosting cannot bin
  — `DropEmptyColumns` drops a column with no value in the *training fold*.

**What phase 3 established.** Modelling goals beats modelling the result on
calibration, and not measurably on anything else. Dixon-Coles (`models/dixon_coles.py`) is
fitted per league — leagues never meet, so a joint fit cannot place their strengths
against each other — once on goals (ρ fitted jointly) and once on each side's xG (a
Poisson quasi-likelihood for the rates, then ρ alone on the actual scorelines, since ρ is
a property of goals). On the same 2025/26 hold-out, same folds:

| | RPS | log loss | Brier |
|---|---|---|---|
| league base rates | 0.2293 | 1.0731 | 0.6489 |
| Elo (k = 30) | 0.2076 | 1.0103 | 0.6046 |
| model #1, form | 0.2091 | 1.0225 | 0.6105 |
| Dixon-Coles on goals (ξ = 0, penalty 4) | 0.2054 | 1.0040 | 0.5995 |
| **Dixon-Coles on xG** (ξ = 0.01, penalty 1) | **0.2045** | **1.0026** | **0.5979** |

- **Neither Dixon-Coles variant can be distinguished from Elo**: RPS difference −0.0023
  [−0.0057, +0.0011] on goals, −0.0031 [−0.0070, +0.0008] on xG. Best point estimates of
  any method, intervals across zero.
- **Goals against the result:** Dixon-Coles on goals beats model #1 on log loss (−0.0185
  [−0.0331, −0.0039]); on RPS the interval ends at +0.0001 — cannot be distinguished,
  only just.
- **xG against goals:** −0.0008 [−0.0039, +0.0024] RPS — cannot be distinguished. On
  validation xG led clearly (0.2031 against 0.2071); on the hold-out the gap all but went.
- **Calibration is where the goals models differ.** Mean |predicted − observed| per class
  (H, D, A): Elo 0.047 / 0.022 / 0.042, model #1 0.052 / 0.052 / 0.047, Dixon-Coles on
  goals 0.027 / 0.013 / 0.027, on xG 0.026 / 0.030 / 0.028. Where Elo says an away win is
  65%, it happens 52%; where Dixon-Coles on goals says 64%, 64%. The overconfidence about
  favourites found in phase 2 belongs to Elo and the form model, not to the data. No
  interval is attached to these gaps.
- **Draws:** Dixon-Coles on goals predicts a 24.9% mean draw rate against 25.5% actual —
  the closest of any model — and still makes a draw the most likely outcome in only 0.3%
  of matches. Being calibrated on draws and backing draws are different things (L11).
- **Short history:** 65 matches, no interval. Dixon-Coles on xG does worst of the models
  there (0.215); a promoted club's xG-fitted strength rests on the three weakest clubs'
  xG, and a direction is all 65 matches can give.

How it was built, and what bit:

- **Targets are now `target_*`.** Model #2 trains on each past match's own xG. Named
  `home_xg`, it would have been picked up by model #1's `home_*`/`away_*` feature selection
  and handed the model this match's xG — L3. `feat_match_outcome` names every target
  `target_*`, so no prefix selection can reach one; a test pins it.
- **Strength is attack *plus* defence.** `d` enters the opponent's rate as `− d`, so a high
  `d` is a good defence. The first version ranked clubs by `a − d` for the promoted-club
  prior, which calls a strong defensive side weak; a test on synthetic seasons caught it,
  and `test_a_promoted_club_starts_at_the_weakest_three_of_last_season` now uses a club
  that the two formulas rank at opposite ends.
- **The gradient is analytic**, including the Dixon-Coles τ terms, and checked against a
  numerical one — about 540 fits per run, which a numerical gradient would make ~45×
  slower.
- **The grid was widened on validation, twice, before the hold-out was touched.** Both
  variants first chose an edge (the xG fit the largest decay, 0.005; the penalties 1 and
  4 at opposite ends). Widened once each way, the optima sat inside the grid, and the
  selection did not change.
- **The hold-out ledger is per method.** Adding a model is not another look at the
  hold-out for the models already scored. Each method records its own evaluations, keyed
  by its own settings; the first report's combined entry was carried over onto base
  rates, Elo and model #1 with its original date, after their scores, calibration, draws
  and all six earlier comparisons reproduced exactly.

**What phase 4 established.** With what Gold carries, an xG model gets a clear step past
distance and angle and stays well short of Understat. `feat_shots` holds the 92,572
non-penalty, non-own-goal shots; model #3 (`models/shot_xg.py`) was chosen on validation
log loss — a logistic regression with splines on distance and angle, situation, body
part, assist, home and minute — and scored once on the 2025/26 hold-out (43,457 shots in
1,752 matches), with intervals from a bootstrap over **matches**:

| | log loss | Brier | AUC | mean predicted (actual 0.0987) |
|---|---|---|---|---|
| distance + angle (baseline) | 0.2912 | 0.0818 | 0.719 | 0.1013 |
| **model #3**, `logistic(C=0.1)` | **0.2802** | **0.0791** | **0.757** | 0.1007 |
| Understat, as published | 0.2589 | 0.0733 | 0.799 | 0.1151 |
| Understat, recalibrated per fold | 0.2563 | 0.0724 | 0.799 | 0.1001 |

- **Beats the baseline:** −0.0111 log loss [−0.0127, −0.0095].
- **Falls short of Understat:** +0.0213 [+0.0188, +0.0237] against it as published, +0.0238
  [+0.0218, +0.0259] against it recalibrated. The gap is discrimination (AUC 0.757 against
  0.799), not calibration.
- **Understat's published xG is miscalibrated on these seasons** — 0.115 per shot against
  0.099 scored, and 0.489 predicted where 0.399 went in for its 0.4–0.6 band. A logistic
  recalibration fitted on each training fold improves it significantly (−0.0026
  [−0.0032, −0.0019]). That is why the recalibrated benchmark is reported: against raw
  Understat, a model calibrated on this data gets some of its score for free.
- **Where the gap sits is consistent with L10.** AUC model / Understat: direct free kicks
  0.581 / 0.590 (log loss even slightly better than recalibrated Understat's), set pieces
  0.770 / 0.800, open play 0.756 / 0.799, corners 0.762 / 0.812. A direct free kick has
  no previous action; open play and corners are where a cross, a through ball or a rebound
  changes the chance. Consistent with, not proof of: there is no interval per situation and
  1,160 free kicks. §13 now recommends raising `last_action` in Part 1.
- **Penalties, as a constant:** 78.9% converted before the hold-out (513), 77.3% in it
  (551), against Understat's 0.756 for every one.
- **Histogram boosting lost to the logistic regression again**, worse with each step of
  capacity, on ~13,000–30,000 training shots.

How it was built, and what bit:

- **The leakage risk for shots is columns, not dates.** A shot's inputs are its own
  attributes at the moment it is taken, so `max_input_date` is null and the date test
  passes trivially. `assert_shot_features_are_on_the_allowlist` fails on any column not
  explicitly approved, and `models/shot_xg.py` lists its inputs explicitly — two guards,
  and a test that the Python list sits inside the dbt one.
- **Two column tests were checking nothing.** The allowlist test, and phase 1's
  `assert_features_come_in_home_away_pairs`, read `information_schema` by table name, so
  dbt saw no dependency and ran them *before* the model was built — on a fresh database
  they found no table and passed; on a rebuild they checked the previous build. The
  planted-leak test for the allowlist found it. Both now read through `ref()` and fail if
  they find no columns at all. **Any test that inspects a model's columns must `ref()` it.**
- **Planted leaks, now eight:** L1, L2, a feature model without the leakage test, a
  one-sided feature, an outcome column in `feat_shots`, penalties let in, own goals let in
  — each asserted to fail on the specific test meant to catch it.
- **Splits and intervals by match (L5).** `splits.aligned` gives shots the match models'
  months; a match's shots share a date, so no match straddles train and test.
  `paired_difference(groups=…)` is a cluster bootstrap — on 10 correlated rows per group
  its interval comes out more than 2.5× wider than resampling rows, which is the
  overconfidence it avoids.

**What phase 5 established** — so far. The machinery is built and proven on synthetic
snapshots; the real test needs time to pass.

- **`football_ml.freeze`** refits all five match methods with the settings validation
  chose (read from `reports/match_outcome.json`, never re-selected) on every match in the
  current snapshot, and writes `models/frozen/<id>/` — the bundle (gitignored) and
  `freeze_card.json` (committed): the moment, the commit, the snapshot, the training range,
  the settings, and the bundle's SHA-256. **It refuses a dirty working tree**: the card's
  commit is only evidence if it contains the code that fitted the model. It also refuses
  when the features no longer match the ones model #1's settings were chosen with.
- **What "frozen" means per method.** Fitted parameters are fixed — Dixon-Coles's club
  strengths, model #1's coefficients, Elo's slope and cut-points. What moves is what is a
  feature: model #1 reads form rebuilt from the later snapshot, and Elo its ratings,
  updated through the new results with each rating from strictly earlier dates (L4).
- **Which matches count.** `scores_matches_after` is the later of the last match fitted
  on and the day of the freeze: a match already played when the model was frozen is not a
  prospective test, even if it was not in the data yet. Tested both ways.
- **`football_ml.prospective`** verifies the bundle's hash against the card before
  anything else, waits (and says so) until the features come from a newer snapshot with
  matches after the freeze, then scores them — paired intervals over matches, draws and
  calibration — into `reports/prospective_<id>.json`.
- **`tests/test_prospective.py`** runs the whole procedure on two synthetic snapshots
  (`synthetic_part2.make_part2(later_rounds=…)`, in which every earlier match is identical
  across the two): freeze on S1, wait, export S2, score exactly the new matches; and the
  refusals — a tampered bundle, a dirty tree, no commit.
- **The predictions page** is Part 2's fifth (decided above): `export_predictions.py`
  writes display rows for the 2025/26 hold-out — what Dixon-Coles on xG, the
  *validation* winner, said before each match, beside Elo — and the prospective matches
  once there are any. Part 2's smoke tests run it in Chromium and Firefox, locally on the
  real export and in CI on a synthetic fixture of the same format.

Two things this phase caught:

- **Double rounding.** Reports stored five decimal places and the export rounded again
  to four, so the page showed 0.2046 for a score of 0.204548 that the README gives as
  0.2045. Reports now store six; a value is rounded once, for display. Re-running both
  trainings with that change moved no result (every difference was 0.000005, a value
  sitting on the old rounding boundary) and left both ledgers untouched.
- **A new field is not a new result.** After phase 4 added `groups` to every comparison,
  the re-run report differed from the committed one in all fourteen comparisons — by that
  one empty field and nothing else. Checked before trusting it.

**To finish phase 5:** commit this phase; run `python -m football_ml.freeze` on the clean
tree and commit the card; after the next matchdays, re-export Part 2's snapshot and
rebuild its marts, rebuild the features here, run `python -m football_ml.prospective`,
and commit the report. The first score will be a few dozen matches — the report says
plainly that its interval is wide.

---

## 13. Open decisions

- **Publishing.** Predictions are derived from Football-Data and Understat data;
  Part 2 §13's licensing question covers them too. `exports/` stays gitignored
  until that is decided.
- ~~**Where predictions are shown.**~~ **Decided in phase 5: a fifth page in Part 2's
  dashboard**, `site/predictions.html`, which copies `exports/predictions/` through
  Part 2's own export (`PREDICTIONS_DIR`). Nothing here writes into Part 2.
- **`last_action` in Gold** (L10). Raise it as a Part 1 task if model #3 shows the
  gap to Understat matters; it would also change Gold's contract in Part 2. **Phase 4
  says it does** (§12): the gap is 0.021 log loss and 0.04 AUC, largest in open play and
  from corners, and gone on direct free kicks — the one situation with no previous action
  to know about. Recommended; not yet raised.
- **Retraining cadence** for prospective evaluation — per snapshot, or frozen for
  a whole season. Frozen is the honest default.

---

## 14. Definition of Done

A piece of work is complete when:

- Every feature comes from a tested dbt model, and the leakage test passes.
- Every model is compared with its baseline, with a paired interval.
- `ruff`, `black`, `pytest` and `dbt build` pass, and notebooks execute top to
  bottom.
- The model card records its training range, snapshot citation, features and git
  commit.
- The README reflects it.

---

## 15. What NOT to do in this repo

- ❌ Do not call Football-Data or Understat, or read Part 1's database directly.
- ❌ Do not write to Part 2's database, or redefine a metric its marts already have.
- ❌ Do not use a current-season summary, or a match's own xG/xPTS, as a feature.
- ❌ Do not use `<=` on dates when building a feature — strictly before.
- ❌ Do not shuffle matches into random folds, or split shots by row.
- ❌ Do not look at the hold-out more than once.
- ❌ Do not turn null xG or missing history into 0.
- ❌ Do not report accuracy as the headline, or claim to beat the betting market.
- ❌ Do not use Understat's xG as a feature of a shot model — it is the benchmark.
- ❌ Do not make causal claims from feature importances.
- ❌ Do not commit `data/`, model binaries, or `exports/` (until §13 decides).
