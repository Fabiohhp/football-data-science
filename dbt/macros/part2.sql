{#
    Where Part 2 is, and where its snapshot's manifest.json is.

    The manifest does not live inside analytics.duckdb: it sits next to the Gold
    snapshot Part 2 built from, which is `$GOLD_SNAPSHOT_DIR` **relative to Part 2's
    root** — the same variable, with the same meaning, that Part 2's own dbt project
    and loaders read. So one setting aims both repositories at the same data, and CI
    sets GOLD_SNAPSHOT_DIR=tests/fixtures/gold once for the pair.
#}

{% macro part2_dir() %}
    {{ return(env_var('PART2_DIR', 'part2_snapshot')) }}
{% endmacro %}

{% macro part2_manifest_path() %}
    {% set snapshot_dir = env_var('GOLD_SNAPSHOT_DIR', 'data/gold') %}
    {% if snapshot_dir.startswith('/') %}
        {{ return(snapshot_dir ~ '/manifest.json') }}
    {% endif %}
    {{ return(part2_dir() ~ '/' ~ snapshot_dir ~ '/manifest.json') }}
{% endmacro %}
