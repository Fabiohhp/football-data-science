"""Per-match predictions, stored for the export to read.

Training and prospective scoring write one row per match they scored — the match,
its result, and every method's H/D/A probabilities — so that
``scripts/export_predictions.py`` can select and format them without fitting or
recomputing anything (CLAUDE.md §4, §9).

They live in ``data/predictions/``, which is gitignored with the rest of ``data/``:
predictions are derived from Football-Data and Understat data, and whether any of
it may be published is still open (CLAUDE.md §13).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from football_ml.evaluate import OUTCOMES

REPO_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_DIR = REPO_ROOT / "data" / "predictions"
MATCH_COLUMNS = [
    "match_key",
    "as_of_date",
    "season",
    "competition_code",
    "home_team_key",
    "away_team_key",
    "target_home_goals",
    "target_away_goals",
    "result",
]


def table(matches: pd.DataFrame, probs: dict[str, np.ndarray]) -> pd.DataFrame:
    """The matches scored, with a ``p_<method>_<H|D|A>`` column per method and outcome."""
    out = matches[MATCH_COLUMNS].reset_index(drop=True).copy()
    for method, p in probs.items():
        for k, outcome in enumerate(OUTCOMES):
            out[f"p_{method}_{outcome}"] = np.asarray(p)[:, k]
    return out


def path(name: str) -> Path:
    return PREDICTIONS_DIR / f"{name}.parquet"


def write(name: str, frame: pd.DataFrame) -> Path:
    target = path(name)
    target.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.register("predictions", frame)
    con.execute(f"COPY predictions TO '{target}' (FORMAT parquet)")
    con.close()
    return target


def read(name: str) -> pd.DataFrame | None:
    target = path(name)
    if not target.exists():
        return None
    return duckdb.connect().execute(f"SELECT * FROM read_parquet('{target}')").df()
