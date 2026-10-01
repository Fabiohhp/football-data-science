"""Model #1 sees the features, never the answer, and treats missing as missing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from football_ml.models.match_outcome import (
    CANDIDATES,
    NOT_FEATURES,
    FormModel,
    feature_columns,
)


def _frame(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    home_form = rng.normal(1.4, 0.5, n)
    away_form = rng.normal(1.4, 0.5, n)
    edge = home_form - away_form + 0.3 + rng.normal(0, 0.6, n)
    result = np.where(edge > 0.4, "H", np.where(edge < -0.3, "A", "D"))
    frame = pd.DataFrame(
        {
            "match_key": range(n),
            "competition_code": rng.choice(["PL", "SA"], n),
            "home_team_key": rng.integers(1, 20, n),
            "away_team_key": rng.integers(1, 20, n),
            "home_form_points_per_match": home_form,
            "away_form_points_per_match": away_form,
            # Nullable integers with pd.NA, as DuckDB returns them, and a column that
            # is empty for the whole frame, as prior_season_* is in 2024/25.
            "home_prior_season_matches": pd.array([pd.NA] * n, dtype="Int64"),
            "away_prior_season_matches": pd.array([pd.NA] * n, dtype="Int64"),
            "target_home_goals": (edge > 0).astype(int) * 2,
            "target_away_goals": (edge < 0).astype(int) * 2,
            "target_home_xg": np.abs(edge),
            "target_away_xg": np.abs(-edge),
            "result": result,
        }
    )
    frame.loc[:9, "home_form_points_per_match"] = np.nan  # cold-start rows
    return frame


def test_the_target_and_identifiers_are_never_features():
    columns = feature_columns(_frame())
    assert not set(columns) & NOT_FEATURES
    assert not [c for c in columns if c.startswith("target_") or c == "result"]
    assert "home_form_points_per_match" in columns


def test_features_come_in_home_away_pairs():
    columns = feature_columns(_frame())
    stems = {c.split("_", 1)[1] for c in columns}
    assert {f"{side}_{stem}" for stem in stems for side in ("home", "away")} == set(columns)


@pytest.mark.parametrize("candidate", CANDIDATES, ids=lambda c: c.name)
def test_every_candidate_fits_with_missing_and_empty_columns(candidate):
    frame = _frame()
    y = np.array([{"H": 0, "D": 1, "A": 2}[r] for r in frame["result"]])
    probs = FormModel(candidate).fit(frame.iloc[:200], y[:200]).predict_proba(frame.iloc[200:])
    assert probs.shape == (100, 3)
    assert probs.sum(axis=1) == pytest.approx(np.ones(100))


def test_a_class_missing_from_training_gets_a_zero_column_not_a_shifted_one():
    frame = _frame()
    y = np.array([{"H": 0, "D": 1, "A": 2}[r] for r in frame["result"]])
    no_draws = y != 1
    model = FormModel(CANDIDATES[2]).fit(frame[no_draws], y[no_draws])
    probs = model.predict_proba(frame.iloc[:5])
    assert (probs[:, 1] == 0).all()
    assert probs.sum(axis=1) == pytest.approx(np.ones(5))


def test_the_model_learns_the_signal_it_is_given():
    frame = _frame(n=2000, seed=1)
    y = np.array([{"H": 0, "D": 1, "A": 2}[r] for r in frame["result"]])
    model = FormModel(CANDIDATES[2]).fit(frame, y)
    strong_home = frame.iloc[[0]].copy()
    strong_home[["home_form_points_per_match", "away_form_points_per_match"]] = [2.8, 0.2]
    weak_home = strong_home.copy()
    weak_home[["home_form_points_per_match", "away_form_points_per_match"]] = [0.2, 2.8]
    assert model.predict_proba(strong_home)[0, 0] > model.predict_proba(weak_home)[0, 0]
