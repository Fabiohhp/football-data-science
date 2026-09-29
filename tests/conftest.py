"""A synthetic Part 2, and a way to run the real ``dbt build`` against it.

Several test files break something on purpose — Part 2's schema, or this
repository's own feature SQL — and assert the build notices. They share this
fixture so each test states only the one thing it breaks.

The synthetic Part 2 is built from ``contracts/part2_columns.json`` itself: every
table the contract lists, with exactly its columns and types, plus one column the
contract does not list (which must not matter). It tracks the contract as it grows.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pytest
from dbt.cli.main import dbtRunner

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((REPO_ROOT / "contracts" / "part2_columns.json").read_text())


def _team_match(match_key, date, team, opponent, is_home, gf, ga, xgf, xga, xpts):
    points = 3 if gf > ga else 1 if gf == ga else 0
    return {
        "match_key": match_key,
        "match_date": date,
        "season": "2025/26",
        "competition_code": "PL",
        "team_key": team,
        "opponent_key": opponent,
        "is_home": is_home,
        "goals_for": gf,
        "goals_against": ga,
        "xg_for": xgf,
        "xg_against": xga,
        "npxg_for": xgf,
        "npxg_against": xga,
        "xpts": xpts,
        "points": points,
    }


# Two fixtures a week apart, so the second has a match of history behind it.
ROWS = {
    "int_team_matches": [
        _team_match(1, "2025-08-16", 10, 20, True, 2, 1, 1.8, 0.9, 2.1),
        _team_match(1, "2025-08-16", 20, 10, False, 1, 2, 0.9, 1.8, 0.6),
        _team_match(2, "2025-08-23", 20, 10, True, 0, 0, 1.1, 1.2, 1.2),
        _team_match(2, "2025-08-23", 10, 20, False, 0, 0, 1.2, 1.1, 1.3),
    ],
    # Team 10 was in the league last season. Its 2025/26 row summarises the season
    # being predicted — the row L2 forbids joining, which a leakage test uses.
    "mart_team_season": [
        {
            "team_key": 10,
            "season": season,
            "competition_code": "PL",
            "played": 38,
            "points_per_match": ppm,
            "matches_with_xg": 38,
            "npxg_for": 60.0,
            "npxg_against": 40.0,
            "is_season_complete": True,
            "last_match_date": last,
        }
        for season, ppm, last in [("2024/25", 2.0, "2025-05-25"), ("2025/26", 2.2, "2026-05-24")]
    ],
}
MATCHES = len({r["match_key"] for r in ROWS["int_team_matches"]})


def make_part2(
    root: Path,
    *,
    matches_in_manifest: int = MATCHES,
    drop_column: str | None = None,
    retype: dict[str, str] | None = None,
    drop_table: str | None = None,
) -> Path:
    """Write a minimal Part 2 checkout — database plus manifest — under ``root``.

    ``drop_column`` and ``retype`` apply to ``int_team_matches``; each keyword
    introduces one defect.
    """
    (root / "data" / "gold").mkdir(parents=True)
    db = root / "data" / "analytics.duckdb"
    con = duckdb.connect(str(db))

    con.execute(f"CREATE TABLE stg_matches AS SELECT range AS match_key FROM range({MATCHES})")

    for table, rows in ROWS.items():
        columns = [(c["name"], c["type"]) for c in CONTRACT["tables"][table]["columns"]]
        if table == "int_team_matches":
            columns = [
                (name, (retype or {}).get(name, type_))
                for name, type_ in columns
                if name != drop_column
            ]
        con.execute(
            f"CREATE TABLE {table} ("
            + ", ".join(f"{name} {type_}" for name, type_ in columns)
            + ")"
        )
        names = [name for name, _ in columns]
        con.executemany(
            f"INSERT INTO {table} VALUES ({', '.join('?' * len(names))})",
            [[row[n] for n in names] for row in rows],
        )
        # A column Part 2 has and this repository does not read must not matter.
        con.execute(f"ALTER TABLE {table} ADD COLUMN something_new_in_part2 INTEGER")

    if drop_table:
        con.execute(f"DROP TABLE {drop_table}")
    con.close()

    manifest = {
        "exported_at": "2026-09-20T12:00:00+00:00",
        "source": "synthetic Part 2 for tests/conftest.py",
        "match_date_range": {"min": "2025-08-16", "max": "2025-08-23"},
        "tables": {
            "fact_matches": {"rows": matches_in_manifest},
            "fact_player_events": {"rows": 0},
            "fact_shots": {"rows": 0},
        },
    }
    (root / "data" / "gold" / "manifest.json").write_text(json.dumps(manifest))
    return db


@dataclass
class BuildResult:
    success: bool
    log: str


@pytest.fixture
def dbt_build(tmp_path, monkeypatch):
    """Run the real ``dbt build`` against the synthetic Part 2 under ``tmp_path/part2``.

    Everything dbt writes — features.duckdb, target/, logs — goes to the temp
    directory, so the tests never touch data/ or the developer's own build. Pass
    ``project_dir`` to build a modified copy of the project (see :func:`copy_project`).
    """
    monkeypatch.chdir(REPO_ROOT)  # the contract path is relative to the repo root
    monkeypatch.setenv("PART2_DIR", str(tmp_path / "part2"))
    monkeypatch.setenv("FEATURES_DUCKDB", str(tmp_path / "features.duckdb"))
    monkeypatch.delenv("GOLD_SNAPSHOT_DIR", raising=False)

    def run(project_dir: Path = REPO_ROOT / "dbt", *, in_subprocess: bool = False) -> BuildResult:
        args = [
            "build",
            "--project-dir", str(project_dir),
            "--profiles-dir", str(REPO_ROOT / "dbt"),
            "--target-path", str(tmp_path / "target"),
            "--log-path", str(tmp_path / "logs"),
            "--no-partial-parse",
        ]  # fmt: skip
        if in_subprocess:
            # dbt-duckdb keeps its connection to features.duckdb open for the life of
            # the process, and DuckDB refuses a second, read-only one beside it. A test
            # that opens the result the way the loader does builds in its own process.
            dbt = Path(sys.executable).parent / "dbt"
            proc = subprocess.run([str(dbt), *args], capture_output=True, text=True)
            return BuildResult(success=proc.returncode == 0, log=proc.stdout + proc.stderr)

        messages: list[str] = []
        result = dbtRunner(callbacks=[lambda event: messages.append(event.info.msg)]).invoke(args)
        log = "\n".join(messages)
        if result.exception is not None:
            log += f"\n{result.exception}"
        return BuildResult(success=result.success, log=log)

    return run


def copy_project(destination: Path) -> Path:
    """A copy of dbt/ without its build output, for a test to modify."""
    return Path(
        shutil.copytree(
            REPO_ROOT / "dbt",
            destination,
            ignore=shutil.ignore_patterns("target", "logs", ".user.yml"),
        )
    )
