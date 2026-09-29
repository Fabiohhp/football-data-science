{#
    The per-team, point-in-time feature columns of feat_team_form — listed once.

    feat_match_outcome takes each of these twice, as home_<name> and away_<name>,
    through `both_sides`, so the model can never see a feature for one side that it
    lacks for the other (CLAUDE.md §6). Adding a feature to feat_team_form means
    adding it here, and then it arrives for both sides or not at all.
#}
{% macro team_form_features() %}
    {{ return([
        'matches_of_history',
        'form_matches',
        'form_prior_season_matches',
        'form_points_per_match',
        'form_goals_for',
        'form_goals_against',
        'form_understat_matches',
        'form_npxg_for',
        'form_npxg_against',
        'form_xpts',
        'season_matches',
        'season_points_per_match',
        'season_understat_matches',
        'season_npxg_difference',
        'prior_season_matches',
        'prior_season_points_per_match',
        'prior_season_npxg_difference',
    ]) }}
{% endmacro %}

{#
    `home.<col> as home_<col>, away.<col> as away_<col>` for every feature, from two
    aliases of feat_team_form joined to the same match.
#}
{% macro both_sides(columns, home_alias='home', away_alias='away') %}
    {%- for col in columns %}
    {{ home_alias }}.{{ col }} as home_{{ col }},
    {{ away_alias }}.{{ col }} as away_{{ col }}{{ ',' if not loop.last }}
    {%- endfor %}
{% endmacro %}
