-- feat_team_form's windows count ROWS ("the 5 matches before this one"), and that
-- means strictly earlier DATES only while no club has two counting matches on one
-- date. match_date has no kickoff time (L4), so a second match that day would sit
-- in the window with the same date as the match being predicted. It holds today;
-- this makes it an assertion rather than an assumption.

select team_key, match_date, count(*) as matches_that_day
from {{ ref('src_team_matches') }}
group by team_key, match_date
having count(*) > 1
