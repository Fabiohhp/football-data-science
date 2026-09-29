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
from pathlib import Path

import duckdb
import yaml

from conftest import MATCHES, REPO_ROOT, make_part2


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
    assert con.execute("SELECT count(*) FROM src_team_matches").fetchone()[0] == 2 * MATCHES
    exported_at, gold_matches = con.execute(
        "SELECT strftime(exported_at AT TIME ZONE 'UTC', '%Y-%m-%dT%H:%M:%S'), gold_matches "
        "FROM src_snapshot"
    ).fetchone()
    assert (exported_at, gold_matches) == ("2026-09-20T12:00:00", MATCHES)


def test_build_refuses_a_part2_stale_against_its_snapshot(tmp_path, dbt_build):
    make_part2(tmp_path / "part2", matches_in_manifest=MATCHES + 1)

    result = dbt_build()

    assert not result.success
    assert "out of date with its snapshot" in result.log
    assert f"stg_matches holds {MATCHES} rows" in result.log
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
    manifest["tables"]["fact_matches"]["rows"] = MATCHES
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
