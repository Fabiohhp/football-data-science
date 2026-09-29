-- The rolling form, recomputed a second, independent way: for each team-match, take
-- that team's matches with match_date STRICTLY before as_of_date, newest first,
-- keep form_window of them, and average the points. A window function and a
-- correlated date filter agreeing on every row is evidence that the window really
-- is "the N matches before", not "the N matches up to and including".
--
-- max_input_date only proves the inputs were early enough; this proves the feature
-- is the number it claims to be.

with recomputed as (

    select
        f.match_key,
        f.team_key,
        f.form_matches,
        f.form_points_per_match,
        r.n       as expected_matches,
        r.ppm     as expected_points_per_match
    from {{ ref('feat_team_form') }} as f,
    lateral (
        select count(*) as n, avg(points) as ppm
        from (
            select t.points
            from {{ ref('src_team_matches') }} as t
            where t.team_key = f.team_key
              and t.match_date < f.as_of_date
            order by t.match_date desc
            limit {{ var('form_window') }}
        )
    ) as r

)

select *
from recomputed
where form_matches <> expected_matches
   or (form_points_per_match is null) <> (expected_points_per_match is null)
   or abs(form_points_per_match - expected_points_per_match) > 1e-9
