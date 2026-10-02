-- Every home_ column of feat_match_outcome has an away_ twin and vice versa, so the
-- model cannot see a feature for one side that it lacks for the other (CLAUDE.md §6).
-- `both_sides` builds them in pairs; this catches a column added by hand beside it.

--
-- Read through ref() so dbt runs it after the model is built; queried by table name,
-- it ran first and checked whatever the previous build had left (see
-- assert_shot_features_are_on_the_allowlist). It also fails if no column is found.
{% set relation = ref('feat_match_outcome') %}

with cols as (
    select column_name
    from duckdb_columns()
    where database_name = '{{ relation.database }}'
      and schema_name = '{{ relation.schema }}'
      and table_name = '{{ relation.identifier }}'
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

union all

select 'feat_match_outcome has no columns: the test found no table to check', []
where (select count(*) from cols) = 0
