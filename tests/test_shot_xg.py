"""Model #3 sees the shot, never its outcome or Understat's xG (L9)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from football_ml.models.shot_xg import (
    BASELINE_FEATURES,
    CANDIDATES,
    FEATURES,
    NOT_INPUTS,
    DistanceAngleBaseline,
    ShotModel,
    Understat,
    UnderstatRecalibrated,
    inputs,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _shots(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    distance = rng.uniform(2, 35, n)
    angle = np.clip(1.2 / distance * 8 + rng.normal(0, 0.05, n), 0.01, 3.0)
    situation = rng.choice(["OpenPlay", "FromCorner", "SetPiece", "DirectFreekick"], n)
    shot_type = rng.choice(["RightFoot", "LeftFoot", "Head", "OtherBodyPart"], n)
    logit = 1.0 - 0.15 * distance + 0.8 * angle - 0.5 * (shot_type == "Head")
    p = 1 / (1 + np.exp(-logit))
    return pd.DataFrame(
        {
            "distance_m": distance,
            "angle_rad": angle,
            "goal_line_m": distance * 0.9,
            "lateral_m": rng.normal(0, 5, n),
            "minute": pd.array(rng.integers(1, 95, n), dtype="Int16"),
            "is_assisted": rng.random(n) < 0.7,
            "is_home": rng.random(n) < 0.5,
            "situation": situation,
            "shot_type": shot_type,
            "benchmark_understat_xg": np.clip(p * 1.15, 0, 1),  # systematically high
            "target_is_goal": rng.random(n) < p,
        }
    )


def test_the_benchmark_and_target_are_never_inputs():
    assert not set(FEATURES) & NOT_INPUTS
    assert not set(BASELINE_FEATURES) & NOT_INPUTS
    assert list(inputs(_shots()).columns) == FEATURES


def test_every_model_input_is_on_the_dbt_allowlist():
    """Two guards on one rule: the Python feature list sits inside the dbt allowlist."""
    sql = REPO_ROOT / "dbt" / "tests" / "assert_shot_features_are_on_the_allowlist.sql"
    allowed = set(re.findall(r"'([a-z_]+)'", sql.read_text().split("%}")[0]))
    assert set(FEATURES) <= allowed


@pytest.mark.parametrize("candidate", CANDIDATES, ids=lambda c: c.name)
def test_every_candidate_fits_and_predicts_probabilities(candidate):
    shots = _shots()
    y = shots["target_is_goal"]
    p = ShotModel(candidate).fit(shots.iloc[:2000], y[:2000]).predict_proba(shots.iloc[2000:])
    assert p.shape == (1000,)
    assert ((p > 0) & (p < 1)).all()


def test_the_baseline_learns_that_closer_is_better():
    shots = _shots(seed=1)
    model = DistanceAngleBaseline().fit(shots, shots["target_is_goal"])
    probe = pd.DataFrame({"distance_m": [5.0, 30.0], "angle_rad": [1.0, 0.2]})
    near, far = model.predict_proba(probe)
    assert near > far


def test_understat_is_passed_through_untouched():
    shots = _shots()
    assert Understat().predict_proba(shots) == pytest.approx(shots["benchmark_understat_xg"])


def test_recalibration_removes_a_systematic_overestimate():
    shots = _shots(n=20_000, seed=2)
    y = shots["target_is_goal"].to_numpy()
    raw = Understat().predict_proba(shots)
    recal = UnderstatRecalibrated().fit(shots, y).predict_proba(shots)
    assert raw.mean() > y.mean() * 1.05  # the planted overestimate
    assert recal.mean() == pytest.approx(y.mean(), rel=0.02)
    # Recalibration is monotone: it keeps Understat's ranking of shots.
    order = np.argsort(raw)
    assert (np.diff(recal[order]) >= -1e-12).all()
