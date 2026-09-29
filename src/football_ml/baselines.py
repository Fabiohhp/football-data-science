"""The two baselines every match-outcome model is reported against (CLAUDE.md §7).

:class:`BaseRates`
    Each league's H/D/A frequencies over the training matches — every match
    strictly before the fold. It knows home advantage and the league's draw rate
    and nothing about either club. A model that cannot beat it has learned nothing.

:func:`pre_match_elo` + :class:`EloForecast`
    A classic Elo rating per club, updated after every match on its result, and an
    ordered logit fitted on the training folds that turns the pre-match rating
    difference into H/D/A probabilities. The bar a form model has to clear.

**Elo is point-in-time by construction.** Ratings are computed by walking the
matches in date order and recording each match's ratings *before* any result of
that date is applied, so a match's rating depends only on strictly earlier dates
(L4) — tested by perturbing results and checking earlier ratings do not move. That
is why the ratings can be computed once over the whole history: a later match
cannot reach back into an earlier rating.

Cold start (L7): every club starts at ``initial`` in the first loaded season, and
a club entering a league afterwards — promoted — starts at the mean final rating
of the ``promoted_from_bottom`` lowest-rated clubs of that league's previous
season: the clubs it is most likely replacing, known before its first match.
There is no between-season regression to the mean: the validation period lies
inside one season, so such a setting could not be tuned there (L6), only guessed.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel

from football_ml.evaluate import OUTCOMES, outcome_index


class BaseRates:
    """Per-league H/D/A frequencies of the training matches."""

    def fit(self, frame: pd.DataFrame, y) -> BaseRates:
        y = np.asarray(y, dtype=int)
        onehot = np.eye(len(OUTCOMES))[y]
        self.overall_ = onehot.mean(axis=0)
        by_league = pd.DataFrame(onehot, index=frame.index).groupby(
            frame["competition_code"].to_numpy()
        )
        self.by_league_ = {league: rows.mean().to_numpy() for league, rows in by_league}
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.vstack(
            [self.by_league_.get(league, self.overall_) for league in frame["competition_code"]]
        )


@dataclass(frozen=True)
class EloSettings:
    k: float = 20.0
    home_advantage: float = 60.0
    initial: float = 1500.0
    promoted_from_bottom: int = 3


DEFAULT_ELO = EloSettings()


def _previous_season(season: str) -> str:
    start = int(season[:4]) - 1
    return f"{start}/{(start + 1) % 100:02d}"


def pre_match_elo(frame: pd.DataFrame, settings: EloSettings = DEFAULT_ELO) -> pd.DataFrame:
    """Both clubs' ratings before each match, aligned to ``frame``'s index.

    Needs ``as_of_date``, ``season``, ``competition_code``, ``home_team_key``,
    ``away_team_key`` and ``result``. Matches are processed a date at a time: every
    rating of a date is read first, then every result of that date applied.
    """
    ratings: dict[int, float] = {}
    clubs_in: dict[tuple[str, str], set[int]] = defaultdict(set)
    home_elo = pd.Series(np.nan, index=frame.index)
    away_elo = pd.Series(np.nan, index=frame.index)
    score = {"H": 1.0, "D": 0.5, "A": 0.0}

    def starting_rating(competition: str, season: str) -> float:
        last_season = clubs_in.get((competition, _previous_season(season)))
        if not last_season:
            return settings.initial
        lowest = sorted(ratings[c] for c in last_season)[: settings.promoted_from_bottom]
        return float(np.mean(lowest))

    for _, day in frame.groupby("as_of_date", sort=True):
        updates: dict[int, float] = defaultdict(float)
        for idx, m in day.iterrows():
            for club in (m.home_team_key, m.away_team_key):
                if club not in ratings:
                    ratings[club] = starting_rating(m.competition_code, m.season)
            h, a = ratings[m.home_team_key], ratings[m.away_team_key]
            home_elo[idx], away_elo[idx] = h, a
            expected = 1.0 / (1.0 + 10 ** (-(h + settings.home_advantage - a) / 400))
            delta = settings.k * (score[m.result] - expected)
            updates[m.home_team_key] += delta
            updates[m.away_team_key] -= delta
        # Only now, with every rating of the day read, do the day's results count.
        for club, delta in updates.items():
            ratings[club] += delta
        for _, m in day.iterrows():
            clubs_in[(m.competition_code, m.season)].update((m.home_team_key, m.away_team_key))

    return pd.DataFrame({"home_elo": home_elo, "away_elo": away_elo})


class EloForecast:
    """An ordered logit from the pre-match Elo difference to H/D/A probabilities.

    One slope and two cut-points, fitted on the training fold. Ordered because the
    outcomes are: a bigger home rating edge moves probability from A through D to H.
    """

    def __init__(self, settings: EloSettings = DEFAULT_ELO):
        self.settings = settings

    def _x(self, elo: pd.DataFrame) -> np.ndarray:
        edge = elo["home_elo"] + self.settings.home_advantage - elo["away_elo"]
        return (edge / 400.0).to_numpy().reshape(-1, 1)

    def fit(self, elo: pd.DataFrame, y) -> EloForecast:
        # OrderedModel wants the categories low to high: A < D < H.
        ordinal = len(OUTCOMES) - 1 - np.asarray(y, dtype=int)
        self.result_ = OrderedModel(ordinal, self._x(elo), distr="logit").fit(
            method="bfgs", disp=False
        )
        return self

    def predict_proba(self, elo: pd.DataFrame) -> np.ndarray:
        low_to_high = np.asarray(self.result_.predict(self._x(elo)))
        return low_to_high[:, ::-1]  # back to H, D, A


def results_to_y(frame: pd.DataFrame) -> np.ndarray:
    return outcome_index(frame["result"])
