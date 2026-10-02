"""Model #3: the chance a shot becomes a goal, from what Gold carries (CLAUDE.md §8 #3).

Question: how close can a location-and-context xG model get to Understat's, with
what Gold carries — location, situation, body part, whether a pass set it up — and
without the action before the shot, which Gold does not have (L10)?

The inputs are listed here explicitly, not picked by prefix: :data:`FEATURES` is the
whole of what a model may see, and :data:`NOT_INPUTS` the columns of feat_shots it
must not — the outcome, and Understat's xG, which is the benchmark (L9). The dbt
allowlist test guards the table; this list guards the model; a test checks the two
never overlap.

Methods:

``baseline``
    Logistic regression on distance and angle alone (CLAUDE.md §7).
``model``
    The candidate chosen on validation from :data:`CANDIDATES`: a logistic regression
    with smooth splines on the geometry and the context one-hot encoded, or histogram
    gradient boosting, which can find interactions — a header's chance falling off
    faster with distance — that the logistic regression cannot.
``understat``
    Understat's own xG, as published. The ceiling (L9), fitted to nothing here.
``understat_recalibrated``
    Understat's xG passed through a logistic recalibration — ``logit(p') = a + b ·
    logit(xG)`` — fitted on each training fold. Understat's xG runs ~15% above the goal
    rate in these seasons; a model calibrated on this data could beat raw Understat on
    log loss through calibration alone. The recalibrated benchmark keeps Understat's
    *ranking* of shots and removes that advantage, so the comparison with it is about
    discrimination. Still the benchmark, never a feature.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, SplineTransformer, StandardScaler

GEOMETRY = ["distance_m", "angle_rad", "goal_line_m", "lateral_m"]
CONTEXT_NUMERIC = ["minute"]
FLAGS = ["is_assisted", "is_home"]
CATEGORICAL = ["situation", "shot_type"]
FEATURES = GEOMETRY + CONTEXT_NUMERIC + FLAGS + CATEGORICAL
BASELINE_FEATURES = ["distance_m", "angle_rad"]

# Columns of feat_shots that are never inputs: the target, and the benchmark (L9).
NOT_INPUTS = frozenset({"target_is_goal", "benchmark_understat_xg"})
SEED = 0
_P_CLIP = 1e-4


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[columns].astype("float64")


def inputs(frame: pd.DataFrame, columns: list[str] = FEATURES) -> pd.DataFrame:
    """The model's view of ``frame``: exactly ``columns``, numerics as float64."""
    numeric = [c for c in columns if c not in CATEGORICAL]
    return _numeric(frame, numeric).join(frame[[c for c in columns if c in CATEGORICAL]])


@dataclass(frozen=True)
class ShotCandidate:
    family: str
    params: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.family}(" + ", ".join(f"{k}={v}" for k, v in self.params.items()) + ")"

    def build(self) -> Pipeline:
        if self.family == "logistic":
            prepare = ColumnTransformer(
                [
                    (
                        "geometry",
                        Pipeline(
                            [
                                ("splines", SplineTransformer(n_knots=6, degree=3)),
                                ("scale", StandardScaler()),
                            ]
                        ),
                        ["distance_m", "angle_rad"],
                    ),
                    ("numeric", StandardScaler(), ["goal_line_m", "lateral_m", "minute"]),
                    ("flags", "passthrough", FLAGS),
                    ("context", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
                ]
            )
            model = LogisticRegression(C=self.params["C"], max_iter=5_000)
        elif self.family == "boosting":
            prepare = ColumnTransformer(
                [
                    ("numeric", "passthrough", GEOMETRY + CONTEXT_NUMERIC + FLAGS),
                    (
                        "context",
                        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                        CATEGORICAL,
                    ),
                ]
            )
            categorical = [False] * len(GEOMETRY + CONTEXT_NUMERIC + FLAGS) + [True] * len(
                CATEGORICAL
            )
            model = HistGradientBoostingClassifier(
                learning_rate=self.params["learning_rate"],
                max_iter=self.params["max_iter"],
                max_leaf_nodes=self.params["max_leaf_nodes"],
                min_samples_leaf=100,
                l2_regularization=1.0,
                categorical_features=categorical,
                early_stopping=False,
                random_state=SEED,
            )
        else:
            raise ValueError(f"unknown shot model family {self.family!r}")
        return Pipeline([("prepare", prepare), ("model", model)])


CANDIDATES = [ShotCandidate("logistic", {"C": c}) for c in (0.01, 0.1, 1.0)] + [
    ShotCandidate("boosting", {"learning_rate": 0.05, "max_iter": it, "max_leaf_nodes": leaves})
    for it in (200, 400)
    for leaves in (15, 31)
]


class ShotModel:
    """A candidate fitted on a training fold; predicts P(goal) per shot."""

    def __init__(self, candidate: ShotCandidate):
        self.candidate = candidate

    def fit(self, frame: pd.DataFrame, y) -> ShotModel:
        self.pipeline_ = self.candidate.build().fit(inputs(frame), np.asarray(y, dtype=int))
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return self.pipeline_.predict_proba(inputs(frame))[:, 1]


class DistanceAngleBaseline:
    """Logistic regression on distance and angle alone."""

    def fit(self, frame: pd.DataFrame, y) -> DistanceAngleBaseline:
        self.pipeline_ = Pipeline(
            [("scale", StandardScaler()), ("model", LogisticRegression(max_iter=5_000))]
        ).fit(_numeric(frame, BASELINE_FEATURES), np.asarray(y, dtype=int))
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return self.pipeline_.predict_proba(_numeric(frame, BASELINE_FEATURES))[:, 1]


def _logit_xg(frame: pd.DataFrame) -> np.ndarray:
    p = np.clip(frame["benchmark_understat_xg"].astype("float64").to_numpy(), _P_CLIP, 1 - _P_CLIP)
    return np.log(p / (1 - p)).reshape(-1, 1)


class Understat:
    """Understat's published xG — the benchmark, fitted to nothing."""

    def fit(self, frame: pd.DataFrame, y) -> Understat:
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return frame["benchmark_understat_xg"].astype("float64").clip(0, 1).to_numpy()


class UnderstatRecalibrated:
    """Understat's xG through ``logit(p') = a + b · logit(xG)``, fitted on the training fold."""

    def fit(self, frame: pd.DataFrame, y) -> UnderstatRecalibrated:
        # Effectively unpenalised: two parameters on tens of thousands of shots.
        self.model_ = LogisticRegression(C=1e6, max_iter=5_000).fit(
            _logit_xg(frame), np.asarray(y, dtype=int)
        )
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(_logit_xg(frame))[:, 1]
