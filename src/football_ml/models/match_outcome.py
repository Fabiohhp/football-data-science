"""Model #1: H/D/A from pre-match form (CLAUDE.md §8).

Question: can pre-match form — goals, xG, xPTS, points — predict H/D/A better than
league base rates and Elo?

The inputs are exactly the ``home_*``/``away_*`` columns of ``feat_match_outcome``,
which dbt built point-in-time and tested for leakage, plus the league. Nothing is
derived here: a feature that is not in the feature layer is not tested there.

Two families of candidate, each a short grid; which one becomes model #1, and with
which settings, is decided on the validation folds only (L6):

``logistic``
    Multinomial logistic regression, L2-regularised; ``C`` is the grid. Missing
    values are imputed with the **training fold's** median, and each imputed column
    gets a missing-indicator beside it — so a club with no history is "median, and
    missing", which the model can weigh, never "zero" (L7).
``boosting``
    scikit-learn's histogram gradient boosting, which takes missing values natively.
    Shallow trees and few iterations: at ~1,500–3,000 training matches, anything
    deeper memorises.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from football_ml.evaluate import OUTCOMES

# Columns with a side prefix that are NOT inputs: identifiers, and the target (L3).
NOT_FEATURES = frozenset({"home_team_key", "away_team_key", "home_goals", "away_goals"})
CATEGORICAL = ["competition_code"]
SEED = 0


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Every paired ``home_*``/``away_*`` feature of ``feat_match_outcome``.

    Read from the frame rather than listed, so a feature added in dbt reaches the
    model without a second edit here — and the exclusions are listed instead, since
    forgetting one of *those* would put the score in the inputs.
    """
    return [c for c in frame.columns if re.match(r"^(home|away)_", c) and c not in NOT_FEATURES]


class DropEmptyColumns(TransformerMixin, BaseEstimator):
    """Drop columns with no observed value in the training fold.

    ``prior_season_*`` is null for every club in 2024/25, the first loaded season, so
    a fold trained only on it has nothing there to learn from — and histogram
    boosting cannot even bin an all-missing column. Which columns are empty is
    decided at fit, on the training fold, like everything else.
    """

    def fit(self, X, y=None):
        self.keep_ = np.flatnonzero(~np.isnan(np.asarray(X, dtype=float)).all(axis=0))
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_]


@dataclass(frozen=True)
class Candidate:
    family: str
    params: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.family}(" + ", ".join(f"{k}={v}" for k, v in self.params.items()) + ")"

    def build(self, numeric: list[str]) -> Pipeline:
        league = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        if self.family == "logistic":
            prepare = ColumnTransformer(
                [
                    (
                        "numeric",
                        Pipeline(
                            [
                                ("drop_empty", DropEmptyColumns()),
                                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                                ("scale", StandardScaler()),
                            ]
                        ),
                        numeric,
                    ),
                    ("league", league, CATEGORICAL),
                ]
            )
            estimator = LogisticRegression(C=self.params["C"], max_iter=5_000)
        elif self.family == "boosting":
            prepare = ColumnTransformer(
                [("numeric", DropEmptyColumns(), numeric), ("league", league, CATEGORICAL)]
            )
            estimator = HistGradientBoostingClassifier(
                learning_rate=self.params["learning_rate"],
                max_iter=self.params["max_iter"],
                max_depth=self.params["max_depth"],
                l2_regularization=1.0,
                min_samples_leaf=40,
                random_state=SEED,
            )
        else:
            raise ValueError(f"unknown model family {self.family!r}")
        return Pipeline([("prepare", prepare), ("model", estimator)])


CANDIDATES = [Candidate("logistic", {"C": c}) for c in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)] + [
    Candidate("boosting", {"learning_rate": 0.05, "max_iter": it, "max_depth": d})
    for it in (100, 200)
    for d in (2, 3)
]


class FormModel:
    """A candidate, fitted on one training fold, predicting H/D/A in that order."""

    def __init__(self, candidate: Candidate):
        self.candidate = candidate

    def _inputs(self, frame: pd.DataFrame) -> pd.DataFrame:
        # DuckDB's nullable integers arrive as pandas Int64 with pd.NA, which
        # scikit-learn refuses. As float64 a missing value is NaN — still missing,
        # never zero (L7).
        return frame[self.features_].astype("float64").join(frame[CATEGORICAL])

    def fit(self, frame: pd.DataFrame, y) -> FormModel:
        self.features_ = feature_columns(frame)
        self.pipeline_ = self.candidate.build(self.features_)
        self.pipeline_.fit(self._inputs(frame), np.asarray(y, dtype=int))
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        proba = self.pipeline_.predict_proba(self._inputs(frame))
        # classes_ are the integer outcomes the fold contained, in sorted order —
        # which is H, D, A — but a column per outcome is guaranteed only if every
        # outcome appeared in training, so place them explicitly.
        out = np.zeros((len(frame), len(OUTCOMES)))
        out[:, self.pipeline_.classes_] = proba
        return out
