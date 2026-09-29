-- One row per counting match: both sides' pre-match features, and the result to
-- predict. as_of_date = match_date.
--
-- Built from two copies of feat_team_form joined to the same match, through the
-- `both_sides` macro, so every feature arrives as a home_/away_ pair (CLAUDE.md §6).
-- The target columns — goals and result — describe the match itself and are never
-- features (L3); they sit at the end, apart from the inputs.

with sides as (

    select
        tm.match_key,
        tm.is_home,
        tm.goals_for,
        form.*
    from {{ ref('src_team_matches') }} as tm
    join {{ ref('feat_team_form') }} as form
        on  form.match_key = tm.match_key
        and form.team_key = tm.team_key

)

select
    home.match_key,
    home.as_of_date,
    home.season,
    home.competition_code,
    home.team_key                                 as home_team_key,
    away.team_key                                 as away_team_key,

    {{ both_sides(team_form_features()) }},

    least(home.matches_of_history, away.matches_of_history) < {{ var('min_history') }}
                                                  as short_history,
    greatest(home.max_input_date, away.max_input_date) as max_input_date,

    -- Target. H/D/A in that order is the ordering the ranked probability score uses.
    home.goals_for                                as home_goals,
    away.goals_for                                as away_goals,
    case
        when home.goals_for > away.goals_for then 'H'
        when home.goals_for = away.goals_for then 'D'
        else 'A'
    end                                           as result

from sides as home
join sides as away
    on  away.match_key = home.match_key
    and not away.is_home
where home.is_home
