"""The committed copy of Part 2 is what a fresh clone builds on, so it has to hold up."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest
import yaml

from football_ml.data import part2_manifest_path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "part2_snapshot" / "data"
CONTRACT = json.loads((REPO_ROOT / "contracts" / "part2_columns.json").read_text())["tables"]


@pytest.fixture
def con():
    connection = duckdb.connect(str(DATA / "analytics.duckdb"), read_only=True)
    yield connection
    connection.close()


def test_the_profile_builds_on_the_committed_copy_by_default():
    profile = yaml.safe_load((REPO_ROOT / "dbt" / "profiles.yml").read_text())
    attachment = profile["football_ml"]["outputs"]["dev"]["attach"][0]

    assert "env_var('PART2_DIR', 'part2_snapshot')" in attachment["path"]


def test_the_loader_checks_against_the_committed_copy_by_default(monkeypatch):
    monkeypatch.delenv("PART2_DIR", raising=False)
    monkeypatch.delenv("GOLD_SNAPSHOT_DIR", raising=False)

    assert part2_manifest_path() == DATA / "gold" / "manifest.json"


def test_the_copy_holds_every_contracted_column_with_its_type(con):
    actual = {
        (table, column): type_
        for table, column, type_ in con.execute(
            "SELECT table_name, column_name, data_type FROM information_schema.columns"
        ).fetchall()
    }

    assert {table for table, _ in actual} == set(CONTRACT)
    for table, spec in CONTRACT.items():
        for column in spec["columns"]:
            assert actual[(table, column["name"])] == column["type"]


def test_the_copy_is_current_with_its_manifest(con):
    manifest = json.loads((DATA / "gold" / "manifest.json").read_text())

    matches = con.execute("SELECT count(*) FROM stg_matches").fetchone()[0]

    assert matches == manifest["tables"]["fact_matches"]["rows"]
