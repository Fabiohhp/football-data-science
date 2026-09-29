-- Every home_ column of feat_match_outcome has an away_ twin and vice versa, so the
-- model cannot see a feature for one side that it lacks for the other (CLAUDE.md §6).
-- `both_sides` builds them in pairs; this catches a column added by hand beside it.

with cols as (
    select column_name
    from information_schema.columns
    where table_name = 'feat_match_outcome'
      and table_catalog = current_database()
),

sided as (
    select
        regexp_extract(column_name, '^(home|away)_', 1)   as side,
        regexp_replace(column_name, '^(home|away)_', '')  as feature
    from cols
    where column_name similar to '(home|away)_.*'
)

select feature, list(side) as sides_present
from sided
group by feature
having count(distinct side) <> 2
