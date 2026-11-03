# Part 2 snapshot

What this repository reads from [Part 2](https://github.com/Fabiohhp/football-data-analytics),
copied here so a clone can reproduce every number with nothing else beside it.

- `data/analytics.duckdb` holds the tables and columns in
  [`contracts/part2_columns.json`](../contracts/part2_columns.json), and nothing more. They are
  the real marts, built from the snapshot described below.
- `data/gold/manifest.json` is that snapshot's manifest: export time, row counts and date range.

`dbt build` reads this folder by default and attaches the database read-only. The contract and
staleness guards check it like any other Part 2. To build on a Part 2 checkout instead, set
`PART2_DIR=../football-data-analytics`. To refresh this copy:

```bash
uv run python scripts/export_part2_snapshot.py ../football-data-analytics
```

Football data provided by the Football-Data.org API. Expected-goals, player and shot data come
from [Understat](https://understat.com/). The data belongs to its providers, under their terms,
and is not covered by this repository's MIT License.
