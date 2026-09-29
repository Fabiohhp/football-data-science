{#
    Refuse a Part 2 database that is stale against its own snapshot — the same check
    Part 2's marts.py makes, for the same reason. The failure it prevents is silent:
    a snapshot re-exported without re-running Part 2's `dbt build` leaves a database
    of last week's matches that answers every query perfectly well, and every feature
    and score here would be built on it while citing the newer manifest.

    stg_matches is 1:1 with Gold's fact_matches, so its row count must equal the
    manifest's.
#}
{% macro assert_part2_is_current() %}
    {% if not execute %}{{ return('') }}{% endif %}

    {% set manifest_path = part2_manifest_path() %}
    {% set result = run_query(
        "select (select count(*) from part2.main.stg_matches), "
        ~ "(select tables.fact_matches.rows from read_json('" ~ manifest_path ~ "'))"
    ) %}
    {% set actual = result.rows[0][0] %}
    {% set expected = result.rows[0][1] %}

    {% if actual != expected %}
        {{ exceptions.raise_compiler_error(
            "Part 2's database is out of date with its snapshot: stg_matches holds "
            ~ actual ~ " rows, but " ~ manifest_path ~ " records " ~ expected
            ~ " matches. Rebuild Part 2 first:\n"
            ~ "  (cd " ~ part2_dir() ~ " && uv run dbt build --project-dir dbt --profiles-dir dbt)"
        ) }}
    {% endif %}
    {{ return('') }}
{% endmacro %}
