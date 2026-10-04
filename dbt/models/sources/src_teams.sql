-- One row per club: its display name, for labelling the predictions export
-- (CLAUDE.md §9). Select and rename only. Part 2 has already unescaped the names
-- (its trap 15), which is why they are read from here rather than from Gold.
--
-- Not a feature: a club's name carries nothing a model should use, and no feat_ model
-- reads this table.

select
    team_key,
    team_name
from {{ source('part2', 'stg_teams') }}
