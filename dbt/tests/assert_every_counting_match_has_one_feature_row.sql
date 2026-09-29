-- feat_match_outcome is built by joining a match's two sides; a lost side drops
-- the match from every training set without an error. One row per counting match,
-- no more and no fewer.

select
    (select count(distinct match_key) from {{ ref('src_team_matches') }}) as counting_matches,
    (select count(*) from {{ ref('feat_match_outcome') }})                as feature_rows
where counting_matches <> feature_rows
