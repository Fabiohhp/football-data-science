"""The build refuses a Part 2 database it should not read from (CLAUDE.md §4).

Two guards run as dbt on-run-start hooks, before any model is built:

- the **column contract** — every Part 2 column in ``contracts/part2_columns.json``
  must exist with its listed type;
- the **staleness check** — Part 2's ``stg_matches`` must hold as many rows as its
  snapshot's ``manifest.json`` records.

A guard that is never seen to fail proves nothing, so each test here builds a small
synthetic Part 2 in a temp directory, breaks it in one specific way, runs the real
``dbt build`` against it and asserts the build stops with a message naming the
problem. The happy path is tested too, including that Part 2's file is byte-for-byte
unchanged afterwards.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pytest
import yaml
from dbt.cli.main import dbtRunner

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((REPO_ROOT / "contracts" / "part2_columns.json").read_text())

# One fixture: Arsenal 2-1 Chelsea, both rows of it, as int_team_matches holds it.
HOME = {
    "match_key": 1,
    "match_date": "2025-08-16",
    "season": "2025/26",
    "competition_code": "PL",
    "team_key": 10,
    "opponent_key": 20,
    "is_home": True,
    "goals_for": 2,
    "goals_against": 1,
    "xg_for": 1.8,
    "xg_against": 0.9,
    "npxg_for": 1.8,
    "npxg_against": 0.9,
    "xpts": 2.1,
    "points": 3,
}
AWAY = HOME | {
    "team_key": 20,
    "opponent_key": 10,
    "is_home": False,
    "goals_for": 1,
    "goals_against": 2,
    "xg_for": 0.9,
    "xg_against": 1.8,
    "npxg_for": 0.9,
    "npxg_against": 1.8,
    "xpts": 0.6,
    "points": 0,
}


def make_part2(
    root: Path,
    *,
    matches_in_manifest: int = 1,
    drop_column: str | None = None,
    retype: dict[str, str] | None = None,
    drop_table: str | None = None,
) -> Path:
    """Write a minimal Part 2 checkout — database plus manifest — under ``root``.

    Built from the contract itself, so it tracks the contract as it grows; the
    keyword arguments each introduce one defect.
    """
    (root / "data" / "gold").mkdir(parents=True)
    db = root / "data" / "analytics.duckdb"
    con = duckdb.connect(str(db))

    con.execute("CREATE TABLE stg_matches AS SELECT 1 AS match_key")

    columns = [
        (c["name"], (retype or {}).get(c["name"], c["type"]))
        for c in CONTRACT["tables"]["int_team_matches"]["columns"]
        if c["name"] != drop_column
    ]
    con.execute(
        "CREATE TABLE int_team_matches ("
        + ", ".join(f"{name} {type_}" for name, type_ in columns)
        + ")"
    )
    names = [name for name, _ in columns]
    con.executemany(
        f"INSERT INTO int_team_matches VALUES ({', '.join('?' * len(names))})",
        [[row[n] for n in names] for row in (HOME, AWAY)],
    )
    # A column Part 2 has and this repository does not read must not matter.
    con.execute("ALTER TABLE int_team_matches ADD COLUMN something_new_in_part2 INTEGER")

    if drop_table:
        con.execute(f"DROP TABLE {drop_table}")
    con.close()

    manifest = {
        "exported_at": "2026-09-20T12:00:00+00:00",
        "source": "synthetic Part 2 for tests/test_part2_guards.py",
        "match_date_range": {"min": "2025-08-16", "max": "2025-08-16"},
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
    directory, so the tests never touch data/ or the developer's own build.
    """
    monkeypatch.chdir(REPO_ROOT)  # the contract path is relative to the repo root
    monkeypatch.setenv("PART2_DIR", str(tmp_path / "part2"))
    monkeypatch.setenv("FEATURES_DUCKDB", str(tmp_path / "features.duckdb"))
    monkeypatch.delenv("GOLD_SNAPSHOT_DIR", raising=False)

    def run() -> BuildResult:
        messages: list[str] = []
        result = dbtRunner(callbacks=[lambda event: messages.append(event.info.msg)]).invoke(
            [
                "build",
                "--project-dir", "dbt",
                "--profiles-dir", "dbt",
                "--target-path", str(tmp_path / "target"),
                "--log-path", str(tmp_path / "logs"),
                "--no-partial-parse",
            ]  # fmt: skip
        )
        log = "\n".join(messages)
        if result.exception is not None:
            log += f"\n{result.exception}"
        return BuildResult(success=result.success, log=log)

    return run


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_succeeds_on_a_current_part2_and_leaves_it_untouched(tmp_path, dbt_build):
    db = make_part2(tmp_path / "part2")
    before = sha256(db)

    result = dbt_build()

    assert result.success, result.log
    assert sha256(db) == before, "the build changed Part 2's database"

    # Not read_only: dbt-duckdb keeps its own connection to this file open inside the
    # test process, and DuckDB refuses a second one with a different configuration.
    con = duckdb.connect(str(tmp_path / "features.duckdb"))
    assert con.execute("SELECT count(*) FROM src_team_matches").fetchone()[0] == 2
    exported_at, gold_matches = con.execute(
        "SELECT strftime(exported_at AT TIME ZONE 'UTC', '%Y-%m-%dT%H:%M:%S'), gold_matches "
        "FROM src_snapshot"
    ).fetchone()
    assert (exported_at, gold_matches) == ("2026-09-20T12:00:00", 1)


def test_build_refuses_a_part2_stale_against_its_snapshot(tmp_path, dbt_build):
    make_part2(tmp_path / "part2", matches_in_manifest=2)

    result = dbt_build()

    assert not result.success
    assert "out of date with its snapshot" in result.log
    assert "stg_matches holds 1 rows" in result.log
    assert "START sql table model" not in result.log, "a model was built from a stale Part 2"


def test_build_refuses_a_column_dropped_from_part2(tmp_path, dbt_build):
    make_part2(tmp_path / "part2", drop_column="xpts")

    result = dbt_build()

    assert not result.success
    assert "int_team_matches.xpts: dropped from Part 2" in result.log


def test_build_refuses_a_column_retyped_in_part2(tmp_path, dbt_build):
    make_part2(tmp_path / "part2", retype={"xg_for": "DECIMAL(8,4)"})

    result = dbt_build()

    assert not result.success
    assert "int_team_matches.xg_for: type changed DOUBLE -> DECIMAL(8,4)" in result.log


def test_build_refuses_a_part2_that_is_not_built(tmp_path, dbt_build):
    make_part2(tmp_path / "part2", drop_table="int_team_matches")

    result = dbt_build()

    assert not result.success
    assert "int_team_matches: in the contract but not in Part 2" in result.log


def test_an_absolute_gold_snapshot_dir_is_honoured(tmp_path, dbt_build, monkeypatch):
    """GOLD_SNAPSHOT_DIR means what it means in Part 2: relative to Part 2's root, or
    absolute. Here the manifest Part 2 would read by default is stale, and the one the
    variable points at is current — so only honouring the variable lets the build pass.
    """
    make_part2(tmp_path / "part2", matches_in_manifest=99)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    manifest = json.loads((tmp_path / "part2" / "data" / "gold" / "manifest.json").read_text())
    manifest["tables"]["fact_matches"]["rows"] = 1
    (elsewhere / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setenv("GOLD_SNAPSHOT_DIR", str(elsewhere))

    assert dbt_build().success


def test_part2_is_attached_read_only():
    """The READ_ONLY attach is the guarantee that nothing here writes to Part 2
    (CLAUDE.md §15) — so the profile is asserted, not just commented."""
    profile = yaml.safe_load((REPO_ROOT / "dbt" / "profiles.yml").read_text())
    attachments = profile["football_ml"]["outputs"]["dev"]["attach"]

    assert [a["alias"] for a in attachments] == ["part2"]
    assert all(a.get("read_only") is True for a in attachments)
