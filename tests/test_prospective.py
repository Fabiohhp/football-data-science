"""Freeze on one snapshot, score on the next — and only on what came after (CLAUDE.md §7).

The whole procedure, end to end, on the synthetic Part 2: build snapshot S1, freeze
the match models on it, see the scorer wait, export a later snapshot S2 with more
rounds of the same season, rebuild, and score — only the matches after the freeze.
Then the ways it must refuse: a tampered bundle, a dirty working tree, no commit.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from conftest import make_part2
from football_ml import data, freeze, prospective
from football_ml.models.match_outcome import feature_columns
from synthetic_part2 import LATER_ROUNDS

S1_EXPORT = "2026-09-20T12:00:00+00:00"
S2_EXPORT = "2026-12-20T12:00:00+00:00"
CLEAN = {"commit": "f" * 40, "dirty": False}


def _report(path):
    """A training report's `selected` block, as train.py writes it — the freeze's input."""
    features = data.open_features()
    predictors = feature_columns(features.table("feat_match_outcome"))
    features.con.close()
    path.write_text(
        json.dumps(
            {
                "selected": {
                    "elo": {"k": 30.0, "home_advantage": 60.0, "initial": 1500.0,
                            "promoted_from_bottom": 3, "name": "elo(k=30)"},
                    "model": {"family": "logistic", "params": {"C": 0.03},
                              "features": predictors, "name": "logistic(C=0.03)"},
                    "dc_goals": {"target": "goals", "xi": 0.0, "penalty": 4.0, "name": "dc"},
                    "dc_xg": {"target": "xg", "xi": 0.01, "penalty": 1.0, "name": "dc"},
                },
                "holdout_evaluations": {},
            }
        )
    )  # fmt: skip
    return path


@pytest.fixture
def frozen_on_s1(tmp_path, dbt_build, monkeypatch):
    """S1 built and the models frozen on it, the moment of the freeze = S1's last match."""
    monkeypatch.setattr(freeze, "git_state", lambda: CLEAN)
    make_part2(tmp_path / "part2", exported_at=S1_EXPORT)
    assert dbt_build(in_subprocess=True).success

    def do_freeze(frozen_at=datetime(2026, 8, 15, 18, 0, tzinfo=UTC)):
        return freeze.freeze(
            frozen_at=frozen_at,
            report_path=_report(tmp_path / "match_outcome.json"),
            frozen_dir=tmp_path / "frozen",
        )

    def export_s2():
        make_part2(tmp_path / "part2", later_rounds=2, exported_at=S2_EXPORT)
        assert dbt_build(in_subprocess=True).success

    return do_freeze, export_s2


def test_the_card_records_what_was_frozen_and_when(frozen_on_s1):
    do_freeze, _ = frozen_on_s1
    card = json.loads(do_freeze().read_text())

    assert card["git"] == CLEAN
    assert card["snapshot"]["exported_at"] == S1_EXPORT
    assert card["training_range"]["last_match"] == "2026-08-15"
    assert card["scores_matches_after"] == "2026-08-15"
    assert set(card["methods"]) == {"base_rates", "elo", "model", "dc_goals", "dc_xg"}
    assert len(card["artifact"]["sha256"]) == 64


def test_on_the_same_snapshot_the_scorer_waits(frozen_on_s1):
    do_freeze, _ = frozen_on_s1
    report = prospective.score(do_freeze(), write=False)
    assert report["status"] == "waiting"
    assert "newer snapshot" in report["reason"]


def test_on_a_later_snapshot_only_the_new_matches_are_scored(frozen_on_s1):
    do_freeze, export_s2 = frozen_on_s1
    card = do_freeze()
    export_s2()

    report = prospective.score(card, write=False)

    assert report["status"] == "scored"
    assert report["matches"] == 4  # two later rounds of a four-club league
    assert report["date_range"] == {"first": LATER_ROUNDS[0], "last": LATER_ROUNDS[1]}
    assert set(report["scores"]) == {"base_rates", "elo", "model", "dc_goals", "dc_xg"}
    for scores in report["scores"].values():
        assert scores["n"] == 4


def test_matches_played_before_the_freeze_are_not_prospective(frozen_on_s1):
    """Frozen after the first later round was played (though not yet in the data):
    that round is not a prospective test, so only the second one is scored."""
    do_freeze, export_s2 = frozen_on_s1
    card = do_freeze(frozen_at=datetime(2026, 10, 20, 9, 0, tzinfo=UTC))
    assert json.loads(card.read_text())["scores_matches_after"] == "2026-10-20"
    export_s2()

    report = prospective.score(card, write=False)

    assert report["matches"] == 2
    assert report["date_range"]["first"] == LATER_ROUNDS[1]


def test_a_bundle_that_is_not_the_frozen_one_is_refused(frozen_on_s1):
    do_freeze, export_s2 = frozen_on_s1
    card = do_freeze()
    export_s2()
    bundle = card.parent / freeze.BUNDLE
    bundle.write_bytes(bundle.read_bytes() + b"\0")

    with pytest.raises(prospective.ProspectiveRefused, match="does not match its freeze card"):
        prospective.score(card, write=False)


def test_a_dirty_tree_cannot_be_frozen(frozen_on_s1, monkeypatch):
    do_freeze, _ = frozen_on_s1
    monkeypatch.setattr(freeze, "git_state", lambda: {"commit": "f" * 40, "dirty": True})
    with pytest.raises(freeze.FreezeRefused, match="Commit first"):
        do_freeze()


def test_without_a_commit_there_is_no_freeze(frozen_on_s1, monkeypatch):
    do_freeze, _ = frozen_on_s1
    monkeypatch.setattr(freeze, "git_state", lambda: {"commit": None, "dirty": None})
    with pytest.raises(freeze.FreezeRefused, match="name the commit"):
        do_freeze()
