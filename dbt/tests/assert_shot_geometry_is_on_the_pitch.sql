-- Distances and angles that a shot from the attacking side of a 105 × 68 pitch can
-- have. A sign error in the coordinate mapping (Understat's x runs towards the goal
-- being attacked, Part 2 trap 10) would put shots behind the goal or 100 m out.

select shot_key, distance_m, angle_rad, goal_line_m
from {{ ref('feat_shots') }}
where distance_m is null or distance_m < 0 or distance_m > 100
   or angle_rad is null or angle_rad < 0 or angle_rad > pi()
   or goal_line_m < 0
