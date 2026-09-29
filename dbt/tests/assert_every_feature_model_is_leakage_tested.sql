-- Every feat_ model must carry the assert_no_feature_reads_the_future test in its
-- schema.yml. A feature model without it could leak and nothing would say so.
--
-- This reads dbt's graph, not a table, so it depends on no model and can never be
-- skipped by a failure upstream — the failure mode that made the leakage test a
-- per-model generic test in the first place.

{% set missing = [] %}
{% if execute %}
    {% set tested = [] %}
    {% for node in graph.nodes.values()
        if node.resource_type == 'test'
        and node.test_metadata is defined and node.test_metadata
        and node.test_metadata.name == 'assert_no_feature_reads_the_future' %}
        {% do tested.extend(node.depends_on.nodes) %}
    {% endfor %}
    {% for node in graph.nodes.values()
        if node.resource_type == 'model' and node.name.startswith('feat_')
        and node.unique_id not in tested %}
        {% do missing.append(node.name) %}
    {% endfor %}
{% endif %}

select model_without_leakage_test
from (values {% for m in missing %}('{{ m }}'),{% endfor %} (null)) as t(model_without_leakage_test)
where model_without_leakage_test is not null
