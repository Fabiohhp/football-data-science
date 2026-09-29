"""The feature loader refuses features built from a snapshot Part 2 no longer holds."""

from __future__ import annotations

import json

import pytest

from conftest import make_part2
from football_ml import data


@pytest.fixture
def built(tmp_path, dbt_build):
    make_part2(tmp_path / "part2")
    result = dbt_build(in_subprocess=True)
    assert result.success, result.log
    return tmp_path


def test_open_features_cites_the_snapshot_in_utc(built):
    features = data.open_features()
    try:
        assert features.snapshot["exported_at"] == "2026-09-20T12:00:00+00:00"
        assert "exported 2026-09-20T12:00:00+00:00" in features.citation()
    finally:
        features.con.close()


def test_features_built_from_an_older_snapshot_are_refused(built):
    manifest_path = built / "part2" / "data" / "gold" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["exported_at"] = "2026-09-27T12:00:00+00:00"  # Part 2 moved on
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(data.FeaturesMissingError, match="Rebuild"):
        data.open_features()


def test_missing_features_name_the_build_command(tmp_path, monkeypatch):
    monkeypatch.setenv("FEATURES_DUCKDB", str(tmp_path / "nowhere.duckdb"))
    with pytest.raises(data.FeaturesMissingError, match="dbt build"):
        data.open_features()
