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

> **Status: phase 0 of 6 — the skeleton.** No model exists yet. What is built is the
> input side: Part 2's database attached read-only, a column contract, and a build
> that refuses stale or drifted input. The design brief, including the leakage traps
> every model here must avoid, is [`CLAUDE.md`](CLAUDE.md).

## Running it

Needs the three repositories side by side and Part 2's marts built.

```bash
uv sync --extra dev
(cd ../football-data-analytics && uv run dbt build --project-dir dbt --profiles-dir dbt)

uv run pytest -q
uv run dbt build --project-dir dbt --profiles-dir dbt    # writes data/features.duckdb
```

Without real data, point both repositories at Part 2's synthetic fixture, as CI does:

```bash
export GOLD_SNAPSHOT_DIR=tests/fixtures/gold   # relative to Part 2's root, in both repos
```
