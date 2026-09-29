-- Part 2's mart_team_season, re-keyed by the season each row may be USED FOR.
--
-- A season summary is computed over every match of that season, so joined to a
-- match inside the same season it hands the model results that had not happened
-- yet (CLAUDE.md §5, L2). Here a 2024/25 row carries `applies_to_season = '2025/26'`,
-- and features join on that column — so the only join that is easy to write is the
-- correct one, and the leakage test checks `last_match_date` besides.
--
-- Select and rename only, plus the re-key. Metrics are Part 2's, unchanged.

select
    team_key,
    competition_code,
    season                          as summarised_season,
    {{ next_season('season') }}     as applies_to_season,
    played,
    points_per_match,
    matches_with_xg,
    npxg_for,
    npxg_against,
    is_season_complete,
    last_match_date
from {{ source('part2', 'mart_team_season') }}
