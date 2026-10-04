"""The predictions export selects and formats; it never computes (CLAUDE.md §9)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_predictions.py"
spec = importlib.util.spec_from_file_location("export_predictions", SCRIPT)
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def _frame():
    rows = []
    for key, result, home, away in [(2, "A", 0, 2), (1, "H", 3, 1)]:
        row = {
            "match_key": key,
            "as_of_date": pd.Timestamp(f"2025-08-1{key}"),
            "season": "2025/26",
            "competition_code": "PL",
            "home_team_key": 10,
            "away_team_key": 20,
            "target_home_goals": home,
            "target_away_goals": away,
            "result": result,
        }
        for method, probs in {"dc_xg": (0.5, 0.3, 0.2), "elo": (0.6, 0.25, 0.15)}.items():
            row |= {f"p_{method}_{o}": p for o, p in zip("HDA", probs, strict=True)}
        rows.append(row)
    return pd.DataFrame(rows)


def test_rows_are_sorted_labelled_and_rounded():
    rows = export._rows(_frame(), {10: "Home FC", 20: "Away FC"})
    assert [r["date"] for r in rows] == ["2025-08-11", "2025-08-12"]
    assert rows[0] | {} == {
        "date": "2025-08-11",
        "league": "PL",
        "home": "Home FC",
        "away": "Away FC",
        "score": "3–1",
        "result": "H",
        "model": [0.5, 0.3, 0.2],
        "baseline": [0.6, 0.25, 0.15],
        "model_on_result": 0.5,
    }


def test_the_probability_on_the_result_is_the_column_the_result_names():
    away_win = export._rows(_frame(), {})[1]
    assert away_win["result"] == "A"
    assert away_win["model_on_result"] == away_win["model"][2] == 0.2


def test_an_unknown_club_is_labelled_not_dropped():
    rows = export._rows(_frame(), {})
    assert rows[0]["home"] == "club 10"


def test_a_comparison_is_relabelled_for_display_with_its_numbers_unchanged():
    raw = [
        {"candidate": "dc_xg", "reference": "elo", "metric": "rps", "value": -0.003051,
         "low": -0.00699, "high": 0.00082, "verdict": "cannot be distinguished", "n": 1752},
        {"candidate": "dc_xg", "reference": "elo", "metric": "log_loss", "value": -0.0077,
         "low": -0.02, "high": 0.004, "verdict": "dc_xg is better", "n": 1752},
    ]  # fmt: skip
    rps = export._comparison(raw, "dc_xg", "elo")
    assert rps == {
        "candidate": "Dixon-Coles on xG",
        "reference": "Elo rating",
        "metric": "rps",
        "difference": -0.0031,
        "low": -0.007,
        "high": 0.0008,
        "verdict": "cannot be distinguished",
        "matches": 1752,
    }
    assert export._comparison(raw, "dc_xg", "elo", "log_loss")["verdict"] == (
        "Dixon-Coles on xG is better"
    )
    assert export._comparison(raw, "model", "elo") is None


@pytest.mark.parametrize("value", [None, 0.123456])
def test_rounding_keeps_missing_as_missing(value):
    assert export._r(value) == (None if value is None else 0.123)
