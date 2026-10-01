"""The hold-out ledger never hides a look at the hold-out (L6)."""

from __future__ import annotations

from football_ml.train import _ledger

GIT = {"commit": "c" * 40, "dirty": False}
ELO = {"k": 30.0, "home_advantage": 60.0, "initial": 1500.0, "promoted_from_bottom": 3}
MODEL = {"family": "logistic", "params": {"C": 0.03}}
FEATURES = ["home_x", "away_x"]
RECORDED = {"base_rates": {}, "elo": ELO, "model": MODEL | {"features": FEATURES}}
FP = {"base_rates": "fb", "elo": "fe", "model": "fm"}
FIRST = {"fingerprint": "combined", "git_commit": "a" * 40, "date": "2026-09-29"}


def _first_report(elo=ELO, model=MODEL, features=FEATURES):
    return {
        "holdout_evaluations": [FIRST],
        "selected": {"elo": elo, "model": model | {"name": "logistic(C=0.03)"}},
        "features": features,
    }


def test_a_first_evaluation_is_recorded_once_per_method():
    ledger = _ledger(None, RECORDED, FP, GIT)
    assert {m: len(v) for m, v in ledger.items()} == {"base_rates": 1, "elo": 1, "model": 1}


def test_reproducing_an_evaluation_adds_nothing():
    first = _ledger(None, RECORDED, FP, GIT)
    again = _ledger({"holdout_evaluations": first}, RECORDED, FP, GIT)
    assert again == first


def test_changed_settings_are_a_second_look():
    first = _ledger(None, RECORDED, FP, GIT)
    again = _ledger({"holdout_evaluations": first}, RECORDED, FP | {"model": "fm2"}, GIT)
    assert len(again["model"]) == 2
    assert len(again["elo"]) == 1


def test_a_new_method_starts_its_own_record():
    first = _ledger(None, RECORDED, FP, GIT)
    again = _ledger(
        {"holdout_evaluations": first}, RECORDED | {"dc_goals": {}}, FP | {"dc_goals": "fd"}, GIT
    )
    assert len(again["dc_goals"]) == 1
    assert again["model"] == first["model"]


def test_the_combined_first_ledger_carries_over_as_the_same_look_when_unchanged():
    ledger = _ledger(_first_report(), RECORDED, FP, GIT)
    for method in ("base_rates", "elo", "model"):
        assert len(ledger[method]) == 1, method
        assert ledger[method][0]["date"] == "2026-09-29"  # the original look, not today


def test_the_combined_first_ledger_is_kept_as_an_earlier_look_when_settings_changed():
    ledger = _ledger(
        _first_report(model={"family": "logistic", "params": {"C": 0.1}}), RECORDED, FP, GIT
    )
    assert [e["date"] for e in ledger["model"]][0] == "2026-09-29"
    assert len(ledger["model"]) == 2
    assert len(ledger["elo"]) == 1


def test_a_changed_feature_list_counts_as_changed_settings():
    ledger = _ledger(_first_report(features=["home_x"]), RECORDED, FP, GIT)
    assert len(ledger["model"]) == 2
