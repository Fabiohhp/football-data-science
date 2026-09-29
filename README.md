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

> **Status: phases 0–2 of 6.** Features, walk-forward splits, two baselines and model #1
> are built. Models #2–#4 are not. The design brief, including the leakage traps every
> model here must avoid, is [`CLAUDE.md`](CLAUDE.md).

## First result: pre-match form does not beat Elo

Question: can a team's recent form (points, goals, non-penalty xG and xPTS over the last
five matches, plus season-to-date and last season) predict home/draw/away better than
league base rates and an Elo rating?

Scored once on the **2025/26 season (1,752 matches)**, walked forward month by month.
Settings were chosen on 2024/25 only. Lower is better:

| | ranked probability score |
|---|---|
| league base rates | 0.2293 |
| **Elo rating** | **0.2076** |
| form model (logistic regression) | 0.2091 |

**The form model and Elo cannot be told apart.** The difference is +0.0015, with a 95%
paired bootstrap interval of −0.0030 to +0.0058. Both beat base rates clearly. Recent
xG adds nothing measurable to a rating that has already absorbed every result. Both
models are also overconfident about favourites. Full numbers, calibration and the caveats
are in [`reports/match_outcome.json`](reports/match_outcome.json).

These are comparisons against the baselines shown. Gold has no betting odds, so nothing
here is compared with, or claims to beat, the market.

## Running it

Needs the three repositories side by side and Part 2's marts built.

```bash
uv sync --extra dev
(cd ../football-data-analytics && uv run dbt build --project-dir dbt --profiles-dir dbt)

uv run pytest -q
uv run dbt build --project-dir dbt --profiles-dir dbt    # writes data/features.duckdb
uv run python -m football_ml.train match_outcome         # report + model card
```

Without real data, point both repositories at Part 2's synthetic fixture, as CI does:

```bash
export GOLD_SNAPSHOT_DIR=tests/fixtures/gold   # relative to Part 2's root, in both repos
```

## CI

CI checks out Part 2 beside this repository, builds Part 2's marts on its synthetic
fixture, then runs lint, tests and `dbt build` here. Part 2 is private, so CI reads it
with a **deploy key**: an SSH key that can read that one repository and nothing else.
To set it up once:

```bash
ssh-keygen -t ed25519 -N "" -C "football-data-science CI" -f part2_deploy_key
```

1. In **football-data-analytics** → Settings → Deploy keys → *Add deploy key*: paste
   `part2_deploy_key.pub`, and leave *Allow write access* unticked.
2. In **football-data-science** → Settings → Secrets and variables → Actions → *New
   repository secret*: name it `PART2_DEPLOY_KEY`, and paste the whole of
   `part2_deploy_key` (the private half).
3. Delete both files locally.
