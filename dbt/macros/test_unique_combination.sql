{#
    Copied from Part 2's dbt/macros/, not imported (CLAUDE.md §6). Every model
    states its grain and tests it, but the grains here are composite — team ×
    match — and dbt's built-in `unique` test only covers a single column.

    Four lines of SQL, vendored rather than pulled in as a dbt_utils dependency,
    so `dbt build` never has to download a package. Use it as:

        tests:
          - unique_combination:
              arguments:
                columns: [match_key, team_key]
#}
{% test unique_combination(model, columns) %}

select
    {{ columns | join(', ') }},
    count(*) as rows_at_this_grain
from {{ model }}
group by {{ range(1, columns | length + 1) | join(', ') }}
having count(*) > 1

{% endtest %}
