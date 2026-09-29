{#
    contracts/part2_columns.json lists every Part 2 column this repository reads,
    with its DuckDB type. This hook fails the build on any difference in those
    columns: a table that is gone, a column that is gone, a column retyped.

    Columns Part 2 has and the contract does not list are ignored on purpose — the
    contract is what *this* repository depends on, not a copy of Part 2's schema,
    and a new Part 2 column breaks nothing here. Nullability is not compared: every
    dbt-built DuckDB table reports every column nullable, so it carries no signal.

    The contract is read through DuckDB's read_text, because dbt's Jinja cannot open
    a file itself. Its path is relative to the repository root (CLAUDE.md §11).
#}
{% macro assert_part2_matches_contract() %}
    {% if not execute %}{{ return('') }}{% endif %}

    {% set raw = run_query("select content from read_text('contracts/part2_columns.json')") %}
    {% set contract = fromjson(raw.rows[0][0]) %}

    {% set live = run_query(
        "select table_name, column_name, data_type from duckdb_columns() "
        ~ "where database_name = 'part2' and schema_name = 'main'"
    ) %}
    {% set actual = {} %}
    {% for row in live.rows %}
        {% do actual.setdefault(row[0], {}).update({row[1]: row[2]}) %}
    {% endfor %}

    {% set problems = [] %}
    {% for table, spec in contract['tables'].items() | sort %}
        {% if table not in actual %}
            {% do problems.append(table ~ ': in the contract but not in Part 2') %}
        {% else %}
            {% for col in spec['columns'] %}
                {% if col['name'] not in actual[table] %}
                    {% do problems.append(table ~ '.' ~ col['name'] ~ ': dropped from Part 2 (contract expects ' ~ col['type'] ~ ')') %}
                {% elif actual[table][col['name']] != col['type'] %}
                    {% do problems.append(table ~ '.' ~ col['name'] ~ ': type changed ' ~ col['type'] ~ ' -> ' ~ actual[table][col['name']]) %}
                {% endif %}
            {% endfor %}
        {% endif %}
    {% endfor %}

    {% if problems %}
        {{ exceptions.raise_compiler_error(
            "Part 2's database at " ~ part2_dir() ~ "/data/analytics.duckdb does not match "
            ~ "contracts/part2_columns.json:\n  - " ~ problems | join('\n  - ')
            ~ "\nIf Part 2 is simply not built, build it first:\n"
            ~ "  (cd " ~ part2_dir() ~ " && uv run dbt build --project-dir dbt --profiles-dir dbt)\n"
            ~ "If Part 2 changed on purpose, update the contract in a deliberate commit and "
            ~ "review every src_ model that reads the column."
        ) }}
    {% endif %}
    {{ return('') }}
{% endmacro %}
