-- Each counting match is two rows, one per side, mirroring each other. Part 2 tests
-- this on int_team_matches; it is re-checked here because every match-level feature
-- will be built by joining a fixture's two rows, and a lone or mismatched row would
-- silently drop or duplicate a match from the training set.

with sides as (
    select
        match_key,
        count(*) as sides,
        count(*) filter (where is_home) as home_sides
    from {{ ref('src_team_matches') }}
    group by match_key
)

select sides.*
from sides
where sides <> 2 or home_sides <> 1

union all

select home.match_key, 2, 1
from {{ ref('src_team_matches') }} as home
join {{ ref('src_team_matches') }} as away
  on away.match_key = home.match_key
 and not away.is_home
where home.is_home
  and (
      home.opponent_key <> away.team_key
      or away.opponent_key <> home.team_key
      or home.goals_for <> away.goals_against
      or home.match_date <> away.match_date
  )
