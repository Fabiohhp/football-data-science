-- One row per shot, from Part 2's mart_shots. Select and rename only (CLAUDE.md §4):
-- penalties and own goals are still here, and feat_shots removes them, so the
-- exclusions are visible in one place and tested there.
--
-- Two columns are not inputs, and the names say so:
--   understat_xg — Understat's own model; the benchmark (L9), never a feature.
--   result       — the shot's outcome ('SavedShot' means on target): an outcome.

select
    shot_key,
    understat_match_id,
    match_key,
    match_date,
    season,
    competition_code,
    team_key,
    is_home,
    assist_player_key,
    minute,
    x,
    y,
    xg              as understat_xg,
    result,
    situation,
    shot_type,
    is_goal,
    is_penalty
from {{ source('part2', 'mart_shots') }}
