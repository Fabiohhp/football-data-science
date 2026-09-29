-- src_snapshot is the one citation every model card and report copies. Zero rows
-- would leave them citing nothing, and two would make "which snapshot?" ambiguous.

select count(*) as rows_in_src_snapshot
from {{ ref('src_snapshot') }}
having count(*) <> 1
