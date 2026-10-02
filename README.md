# Football data — Part 3: Machine Learning

Models that predict what has not happened yet — match outcomes, and the chance a
shot becomes a goal — evaluated against baselines with intervals, on splits that
could not have seen the future.

The last part of a three-repository portfolio:

| Part | Repository | |
|------|------------|---|
| 1 — Data Engineering | [`football-data-engineering`](../football-data-engineering) | the pipeline; owns the Gold layer |
| 2 — Analytics & BI | [`football-data-analytics`](../football-data-analytics) | metrics layer, analyses, dashboard |
| **3 — Machine Learning** | **this repository** | models and their evaluation |

> **Status: phases 0–4 of 6.** Features, walk-forward splits, baselines, two match
> models and a shot xG model are built. Prospective scoring is not built yet. The design
> brief, including the leakage traps every model here must avoid, is
> [`CLAUDE.md`](CLAUDE.md).

## Results so far: nothing beats Elo measurably, but goals models are better calibrated

Scored once on the **2025/26 season (1,752 matches)**, walked forward month by month.
Each model's settings were chosen on 2024/25 only. Lower is better:

| | ranked probability score | log loss |
|---|---|---|
| league base rates | 0.2293 | 1.0731 |
| Elo rating | 0.2076 | 1.0103 |
| form model (logistic regression on last-5 form, xG, xPTS) | 0.2091 | 1.0225 |
| Dixon-Coles, fitted to goals | 0.2054 | 1.0040 |
| Dixon-Coles, fitted to xG | **0.2045** | **1.0026** |

- **Against Elo, no model is measurably better.** Every 95% paired bootstrap interval of
  the difference straddles zero. For Dixon-Coles on xG it is −0.0070 to +0.0008.
- **Modelling goals beats modelling the result on log loss.** Dixon-Coles on goals against
  the form model: −0.0185, interval −0.0331 to −0.0039.
- **Fitting to xG instead of goals** cannot be told apart on the hold-out, although it
  led clearly on validation.
- **The goals models are calibrated; Elo and the form model are not.** Where Elo gives
  an away win 65%, it happens 52% of the time. Where Dixon-Coles gives 64%, it happens 64%.

Full numbers, calibration tables and caveats are in
[`reports/match_outcome.json`](reports/match_outcome.json) and CLAUDE.md §12.

These are comparisons against the baselines shown. Gold has no betting odds, so nothing
here is compared with, or claims to beat, the market.

## The shot model: past distance and angle, short of Understat

How close can an xG model get to Understat's using what Gold carries: location,
situation, body part, assist? Scored once on 2025/26 (43,457 non-penalty shots).
Intervals come from resampling matches, not shots. Lower log loss is better; higher AUC
means better at telling goals from misses:

| | log loss | AUC |
|---|---|---|
| distance + angle | 0.2912 | 0.719 |
| **this model** | **0.2802** | **0.757** |
| Understat's xG, recalibrated | 0.2563 | 0.799 |

- **Better than distance and angle:** −0.0111, interval −0.0127 to −0.0095.
- **Clearly short of Understat:** +0.0238, interval +0.0218 to +0.0259.
- **The gap vanishes on direct free kicks**, the one situation with no action before the
  shot. It is widest in open play and from corners. That is consistent with what Gold
  lacks: the pass or cross before the shot, which Understat's model uses. It is a reason
  to add that column upstream, not a proof that it is the whole gap.
- **Understat's published xG runs high on these seasons:** 0.115 per shot against 0.099
  scored. The comparison is therefore made against it recalibrated, so the model doesn't
  win on calibration it got for free.

## Running it

Needs the three repositories side by side and Part 2's marts built.

```bash
uv sync --extra dev
(cd ../football-data-analytics && uv run dbt build --project-dir dbt --profiles-dir dbt)

uv run pytest -q
uv run dbt build --project-dir dbt --profiles-dir dbt    # writes data/features.duckdb
uv run python -m football_ml.train match_outcome         # report + model cards
uv run python -m football_ml.train shot_xg               # the shot model
```

Without real data, point both repositories at Part 2's synthetic fixture, as CI does:

```bash
export GOLD_SNAPSHOT_DIR=tests/fixtures/gold   # relative to Part 2's root, in both repos
```

## CI

Two jobs. **`build`** needs nothing outside this repository. It generates a synthetic
Part 2 from the column contract (`tests/synthetic_part2.py`), then runs lint, tests
and `dbt build` on top of it. **`part2-integration`** checks out the real Part 2
beside this repository, builds Part 2's marts on Part 2's own fixture, and runs
`dbt build` here. That catches a Part 2 change that would break this repository.

Part 2 is private, so the second job reads it with a **deploy key**: an SSH key that
can read that one repository and nothing else. Without the key, the job reports that it
was skipped and passes. To set it up once:

```bash
ssh-keygen -t ed25519 -N "" -C "football-data-science CI" -f part2_deploy_key
```

1. In **football-data-analytics** → Settings → Deploy keys → *Add deploy key*: paste
   `part2_deploy_key.pub`, and leave *Allow write access* unticked.
2. In **football-data-science** → Settings → Secrets and variables → Actions → *New
   repository secret*: name it `PART2_DEPLOY_KEY`, and paste the whole of
   `part2_deploy_key` (the private half).
3. Delete both files locally.
