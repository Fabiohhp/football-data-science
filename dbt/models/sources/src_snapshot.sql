-- One row: the Gold snapshot behind Part 2's database, copied from its
-- manifest.json at build time.
--
-- The manifest does not live inside analytics.duckdb, so without this table
-- features.duckdb could not say which data it was built from. With it, the database
-- carries its own provenance: every model card and report cites this row (CLAUDE.md
-- §4), and needs neither Part 2's checkout nor the manifest file to do so.
--
-- Read as text and extracted by path, not through read_json: read_json infers
-- `exported_at` as a naive TIMESTAMP and drops its `+00:00`, and casting that to
-- timestamptz then reads it as *local* time — a citation three hours out on a
-- machine in UTC-3 and correct in CI, which runs in UTC. Casting the original string
-- keeps the offset. tests/test_part2_guards.py pins it.

with manifest as (
    select content::json as m
    from read_text('{{ part2_manifest_path() }}')
)

select
    cast(m ->> '$.exported_at' as timestamptz)                  as exported_at,
    cast(m ->> '$.match_date_range.min' as date)                as match_date_min,
    cast(m ->> '$.match_date_range.max' as date)                as match_date_max,
    cast(m ->> '$.tables.fact_matches.rows' as bigint)          as gold_matches,
    cast(m ->> '$.tables.fact_player_events.rows' as bigint)    as gold_player_events,
    cast(m ->> '$.tables.fact_shots.rows' as bigint)            as gold_shots,
    m ->> '$.source'                                            as gold_source
from manifest
