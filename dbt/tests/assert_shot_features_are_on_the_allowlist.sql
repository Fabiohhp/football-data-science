-- The leakage guard for shots is about columns (see feat_shots): a shot's outcome
-- (`result`) or Understat's xG (L9) must never become an input. Every column of
-- feat_shots must be on this list, and a new one fails the build until it is added
-- here — a reviewed decision, with the reason it is known before the shot.

{% set allowed = [
    'shot_key', 'match_key', 'understat_match_id', 'as_of_date', 'season',
    'competition_code',
    'distance_m', 'angle_rad', 'goal_line_m', 'lateral_m', 'situation', 'shot_type',
    'is_assisted', 'is_home', 'minute',
    'max_input_date',
    'benchmark_understat_xg', 'target_is_goal',
] %}

{#
    Read through ref(), not by table name. The first version queried
    information_schema for 'feat_shots' directly; dbt then saw no dependency, ran this
    test BEFORE the model was built, and passed on a fresh database because there was
    no table to find columns in (tests/test_leakage.py caught it). The ref() orders
    the test after the model, and the second branch fails if no column is found at
    all, so the test can never pass by looking at nothing.
#}
{% set relation = ref('feat_shots') %}

with cols as (
    select column_name
    from duckdb_columns()
    where database_name = '{{ relation.database }}'
      and schema_name = '{{ relation.schema }}'
      and table_name = '{{ relation.identifier }}'
)

select column_name as problem
from cols
where column_name not in ({% for c in allowed %}'{{ c }}'{{ ',' if not loop.last }}{% endfor %})

union all

select 'feat_shots has no columns: the test found no table to check'
where (select count(*) from cols) = 0
