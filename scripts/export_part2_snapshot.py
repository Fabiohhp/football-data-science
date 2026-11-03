"""Copy what this repository reads from Part 2 into part2_snapshot/.

    uv run python scripts/export_part2_snapshot.py ../football-data-analytics

A clone of this repository then builds on real data by itself. The copy holds only the tables and
columns in contracts/part2_columns.json, plus the manifest of the Gold snapshot they were built
from, so the on-run-start guards check it like any other Part 2. Re-run it, and commit the result,
when Part 2's marts change.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((REPO_ROOT / "contracts" / "part2_columns.json").read_text())
OUT = REPO_ROOT / "part2_snapshot" / "data"


def export(part2_dir: Path, out: Path = OUT) -> Path:
    """Write ``out/analytics.duckdb`` and ``out/gold/manifest.json`` from a built Part 2."""
    (out / "gold").mkdir(parents=True, exist_ok=True)
    target = out / "analytics.duckdb"
    target.unlink(missing_ok=True)

    con = duckdb.connect(str(target))
    con.execute(f"ATTACH '{part2_dir / 'data' / 'analytics.duckdb'}' AS part2 (READ_ONLY)")
    for table, spec in CONTRACT["tables"].items():
        # stg_matches is only counted (assert_part2_is_current), so it lists no columns.
        columns = [column["name"] for column in spec["columns"]] or ["match_key"]
        con.execute(f"CREATE TABLE {table} AS SELECT {', '.join(columns)} FROM part2.main.{table}")
    con.close()

    shutil.copy(part2_dir / "data" / "gold" / "manifest.json", out / "gold" / "manifest.json")
    return target


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: export_part2_snapshot.py PATH_TO_PART2")
    print(f"wrote {export(Path(sys.argv[1]))}")
