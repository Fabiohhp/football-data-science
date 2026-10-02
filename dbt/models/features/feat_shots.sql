-- One row per non-penalty, non-own-goal shot: what was known at the moment it was
-- taken, and whether it went in. The input to model #3 (CLAUDE.md §8 #3).
--
-- Excluded, and tested to be excluded (assert_feat_shots_has_every_eligible_shot):
--   own goals — rows in fact_shots at the scorer's own end with 0.0 xG; not shots
--               (Part 2 trap 11);
--   penalties — all at one spot with one xG (trap 10): a location model would learn
--               them perfectly and flatter its score. Reported separately instead.
--
-- THE LEAKAGE RISK HERE IS COLUMNS, NOT DATES. The as-of moment is the shot itself,
-- and its inputs are its own attributes at that moment — where it was taken from, the
-- situation, the body part, whether a pass set it up. No input comes from another row,
-- so max_input_date is null and the date-based leakage test passes trivially. What
-- would leak is a column: `result` ('SavedShot' means on target) or Understat's xG
-- (L9). assert_shot_features_are_on_the_allowlist fails on any column not explicitly
-- approved, so adding one is a reviewed decision.
--
-- What is NOT here, because Gold does not carry it (L10): the action before the shot
-- (cross, through ball, rebound), which Understat's own model uses.

with shots as (

    select
        *,
        (1 - x) * {{ var('pitch_length_m') }}   as goal_line_m,
        (y - 0.5) * {{ var('pitch_width_m') }}  as lateral_m
    from {{ ref('src_shots') }}
    where not is_penalty
      and result <> 'OwnGoal'

)

select
    shot_key,
    match_key,
    understat_match_id,
    match_date                                    as as_of_date,
    season,
    competition_code,

    -- Features: the shot at the moment it was taken.
    sqrt(goal_line_m * goal_line_m + lateral_m * lateral_m) as distance_m,
    -- The angle the goal mouth subtends from the shot: π when level with the goal
    -- line between the posts, near 0 far out or wide. atan2 keeps it in [0, π].
    atan2(
        {{ var('goal_width_m') }} * goal_line_m,
        goal_line_m * goal_line_m + lateral_m * lateral_m
            - ({{ var('goal_width_m') }} / 2) * ({{ var('goal_width_m') }} / 2)
    )                                             as angle_rad,
    goal_line_m,
    lateral_m,
    situation,
    shot_type,
    assist_player_key is not null                 as is_assisted,
    is_home,
    minute,

    cast(null as date)                            as max_input_date,

    -- Not inputs. The benchmark (L9) and the target.
    understat_xg                                  as benchmark_understat_xg,
    is_goal                                       as target_is_goal

from shots
