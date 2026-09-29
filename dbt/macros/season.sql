{#
    Season labels are Part 1's 'YYYY/YY' (Part 2 trap 5). The season after
    '2024/25' is '2025/26'; the one after '2099/00' is '2100/01'.
#}
{% macro next_season(season) %}
    (
        cast(cast(left({{ season }}, 4) as integer) + 1 as varchar)
        || '/'
        || lpad(cast((cast(right({{ season }}, 2) as integer) + 1) % 100 as varchar), 2, '0')
    )
{% endmacro %}
