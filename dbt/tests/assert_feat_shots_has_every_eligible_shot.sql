-- feat_shots holds every shot that is neither a penalty nor an own goal — no more
-- (a penalty or own goal let through) and no fewer (a shot lost by a filter).

select
    (select count(*) from {{ ref('src_shots') }}
      where not is_penalty and result <> 'OwnGoal')                 as eligible,
    (select count(*) from {{ ref('feat_shots') }})                  as in_features,
    (select count(*) from {{ ref('feat_shots') }}
      where situation = 'Penalty')                                  as penalties_let_through
where eligible <> in_features or penalties_let_through > 0
