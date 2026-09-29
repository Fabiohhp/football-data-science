-- One row per team per counting match: what was known about that team BEFORE the
-- match. as_of_date = match_date; every input is from a strictly earlier date.
--
-- Three kinds of history, each with its own coverage column:
--
--   form_*          the last `form_window` matches. CARRIES OVER season boundaries:
--                   in August it still holds May's matches, and
--                   form_prior_season_matches says how many of them are.
--   season_*        season to date. RESETS each season.
--   prior_season_*  the previous season's summary from Part 2's mart_team_season,
--                   in the same competition (L2). Null for a club that was not in
--                   the league last season — promoted, or any club in 2024/25,
--                   the first loaded season.
--
-- Nothing is imputed. A club with no earlier match has null form, not zero (L7), and
-- xG averages skip matches the sources never paired (Part 2 trap 7), counted in
-- *_understat_matches — xG, npxG and xPTS are always null together, so one count
-- covers all three.
--
-- THE WINDOW FRAMES END AT `1 preceding`, NOT `current row` (L1). That is the whole
-- difference between a feature and the answer. It is strict in time, not just in
-- rows, because no club plays twice on one date — asserted by
-- assert_no_club_plays_twice_in_a_day, since the frame counts rows, not days.
-- assert_no_feature_reads_the_future fails if either stops being true.

with team_matches as (

    select * from {{ ref('src_team_matches') }}

),

windows as (

    select
        match_key,
        team_key,
        match_date,
        season,
        competition_code,

        count(*) over all_before                  as matches_of_history,
        max(match_date) over all_before           as history_last_input_date,

        count(*) over form                        as form_matches,
        avg(points) over form                     as form_points_per_match,
        avg(goals_for) over form                  as form_goals_for,
        avg(goals_against) over form              as form_goals_against,
        count(npxg_for) over form                 as form_understat_matches,
        avg(npxg_for) over form                   as form_npxg_for,
        avg(npxg_against) over form               as form_npxg_against,
        avg(xpts) over form                       as form_xpts,
        max(match_date) over form                 as form_last_input_date,

        count(*) over season_to_date              as season_matches,
        avg(points) over season_to_date           as season_points_per_match,
        count(npxg_for) over season_to_date       as season_understat_matches,
        avg(npxg_for - npxg_against) over season_to_date as season_npxg_difference,
        max(match_date) over season_to_date       as season_last_input_date

    from team_matches
    window
        all_before as (
            partition by team_key order by match_date
            rows between unbounded preceding and 1 preceding
        ),
        form as (
            partition by team_key order by match_date
            rows between {{ var('form_window') }} preceding and 1 preceding
        ),
        season_to_date as (
            partition by team_key, season order by match_date
            rows between unbounded preceding and 1 preceding
        )

)

select
    w.match_key,
    w.team_key,
    w.match_date                                  as as_of_date,
    w.season,
    w.competition_code,

    w.matches_of_history,

    w.form_matches,
    -- The window holds the most recent matches, so the ones from this season are the
    -- first min(form_matches, season_matches) of them and the rest are earlier.
    w.form_matches - least(w.form_matches, w.season_matches) as form_prior_season_matches,
    w.form_points_per_match,
    w.form_goals_for,
    w.form_goals_against,
    w.form_understat_matches,
    w.form_npxg_for,
    w.form_npxg_against,
    w.form_xpts,

    w.season_matches,
    w.season_points_per_match,
    w.season_understat_matches,
    w.season_npxg_difference,

    prior.played                                  as prior_season_matches,
    prior.points_per_match                        as prior_season_points_per_match,
    (prior.npxg_for - prior.npxg_against)
        / nullif(prior.matches_with_xg, 0)        as prior_season_npxg_difference,

    -- The latest date any input on this row came from. Null only when there is no
    -- input at all. The leakage test requires it to be strictly before as_of_date.
    greatest(
        w.history_last_input_date,
        w.form_last_input_date,
        w.season_last_input_date,
        prior.last_match_date
    )                                             as max_input_date

from windows as w
left join {{ ref('src_team_season_prior') }} as prior
    on  prior.team_key = w.team_key
    and prior.competition_code = w.competition_code
    and prior.applies_to_season = w.season
