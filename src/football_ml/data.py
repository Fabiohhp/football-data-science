"""Reading the feature layer.

Training reads ``data/features.duckdb``, never Part 2 directly: the features are
where leakage is prevented and tested (CLAUDE.md §4), so a model that read around
them would skip the test the repository rests on.

    from football_ml import data

    f = data.open_features()
    frame = f.df("SELECT * FROM feat_match_outcome")
    print(f.citation())

**It refuses a feature database built from a different snapshot than Part 2 holds
now.** ``src_snapshot`` records the manifest the features were built from; if
Part 2 has since been rebuilt on a newer export, the features are last week's and a
model trained on them would cite the wrong data. The same silent failure Part 2's
``marts.py`` guards against, one layer further down.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURES_PATH = REPO_ROOT / "data" / "features.duckdb"
BUILD_COMMAND = "uv run dbt build --project-dir dbt --profiles-dir dbt"


class FeaturesMissingError(RuntimeError):
    """The feature layer is not built, or is out of date with Part 2's snapshot.

    Carries the command that fixes it.
    """


def _repo_path(value: str) -> Path:
    """A path setting, resolved against the repository root rather than the cwd.

    Part 2's rule and reason: pytest runs from the root and a notebook kernel from
    ``notebooks/``, so resolving against the cwd would make one setting mean two
    places and fail only in the notebook.
    """
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def features_path() -> Path:
    """``FEATURES_DUCKDB`` if set, else ``data/features.duckdb`` — as dbt's profile."""
    return _repo_path(os.environ.get("FEATURES_DUCKDB", str(DEFAULT_FEATURES_PATH)))


def part2_manifest_path() -> Path:
    """Where Part 2's current snapshot manifest is — the same rule as dbt's macro.

    ``GOLD_SNAPSHOT_DIR`` is Part 2's variable and keeps Part 2's meaning: relative
    to Part 2's root, or absolute.
    """
    part2 = _repo_path(os.environ.get("PART2_DIR", "../football-data-analytics"))
    snapshot = Path(os.environ.get("GOLD_SNAPSHOT_DIR", "data/gold")).expanduser()
    return (snapshot if snapshot.is_absolute() else part2 / snapshot) / "manifest.json"


@dataclass(frozen=True)
class Features:
    """A read-only connection to the built feature layer, and its provenance."""

    path: Path
    snapshot: dict[str, Any]
    con: duckdb.DuckDBPyConnection

    def df(self, query: str) -> pd.DataFrame:
        return self.con.sql(query).df()

    def table(self, name: str) -> pd.DataFrame:
        return self.df(f"SELECT * FROM {name}")

    def citation(self) -> str:
        """The line every report and model card carries, naming the data."""
        s = self.snapshot
        return (
            f"features built from the Gold snapshot exported {s['exported_at']}, "
            f"matches {s['match_date_min']} to {s['match_date_max']} "
            f"({s['gold_matches']:,} matches, {s['gold_shots']:,} shots)"
        )


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value


def open_features(path: Path | None = None, *, check_part2: bool = True) -> Features:
    """Open the feature layer read-only, and refuse it if Part 2 has moved on.

    ``check_part2=False`` skips the comparison with Part 2's current manifest. It
    exists for tests that build a feature database on purpose; training should never
    pass it.
    """
    path = path or features_path()
    if not path.exists():
        raise FeaturesMissingError(f"No feature layer at {path}. Build it:\n  {BUILD_COMMAND}")

    con = duckdb.connect(database=str(path), read_only=True)
    # Timestamps come back in UTC whatever the machine's zone, so a citation reads
    # the same on a laptop in UTC-3 and in CI.
    con.execute("SET TimeZone = 'UTC'")
    try:
        cursor = con.execute("SELECT * FROM src_snapshot")
        names = [d[0] for d in cursor.description]
        snapshot = {n: _jsonable(v) for n, v in zip(names, cursor.fetchone(), strict=True)}
    except duckdb.Error as exc:
        con.close()
        raise FeaturesMissingError(
            f"{path} has no src_snapshot table. Rebuild:\n  {BUILD_COMMAND}"
        ) from exc

    if check_part2:
        manifest_path = part2_manifest_path()
        if not manifest_path.exists():
            con.close()
            raise FeaturesMissingError(
                f"Cannot check the features against Part 2: no manifest at {manifest_path}."
            )
        manifest = json.loads(manifest_path.read_text())
        built_from = datetime.fromisoformat(snapshot["exported_at"])
        current = datetime.fromisoformat(manifest["exported_at"])
        if built_from != current:
            con.close()
            raise FeaturesMissingError(
                f"The features were built from the snapshot exported {built_from}, but "
                f"Part 2 now holds the one exported {current}. Rebuild:\n  {BUILD_COMMAND}"
            )

    return Features(path=path, snapshot=snapshot, con=con)
