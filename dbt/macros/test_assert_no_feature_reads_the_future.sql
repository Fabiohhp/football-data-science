{#
    THE test this repository rests on (CLAUDE.md §6): no feature row may use an input
    from its own as-of date or later. Every feat_ model carries `as_of_date` (the
    moment it predicts from) and `max_input_date` (the latest date any of its inputs
    came from); a row where max_input_date >= as_of_date has seen the match it is
    about, or a same-day match (L4), and fails.

    Null max_input_date means the row used no input at all — a club's first loaded
    match — and passes: nothing was read, so nothing was read from the future.

    **A generic test attached to each feat_ model, not one singular test over all of
    them.** It was first written as a singular test reading every feat_ model, and
    the test proving it has teeth (tests/test_leakage.py) showed it being *skipped*:
    `dbt build` skips everything downstream of a failing test, so when another test
    on feat_team_form failed first, feat_match_outcome was never built and a test that
    also read it never ran. Attached per model, it runs as soon as that model exists.
    `assert_every_feature_model_is_leakage_tested` makes sure no feat_ model lacks it.

        data_tests:
          - assert_no_feature_reads_the_future
#}
{% test assert_no_feature_reads_the_future(model) %}

select
    as_of_date,
    max_input_date,
    count(*) as rows_reading_the_future
from {{ model }}
where max_input_date >= as_of_date
group by all

{% endtest %}
