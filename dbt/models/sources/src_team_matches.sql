-- One row per team per counting match, straight from Part 2's int_team_matches.
-- Select and rename only (CLAUDE.md §4): every column here is listed in
-- contracts/part2_columns.json, and nothing is filtered or derived.
--
-- Every value on a row describes *that* match — its goals, its xG, its xPTS — so
-- none of it is a feature of that match (L3). feat_ models may only use rows from
-- strictly earlier dates (L4).

select
    match_key,
    match_date,
    season,
    competition_code,
    team_key,
    opponent_key,
    is_home,
    goals_for,
    goals_against,
    xg_for,
    xg_against,
    npxg_for,
    npxg_against,
    xpts,
    points
from {{ source('part2', 'int_team_matches') }}
