"""Base rates and Elo: correct updates, and no rating that knows the future (L4, L7)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from football_ml import evaluate as ev
from football_ml.baselines import BaseRates, EloForecast, EloSettings, pre_match_elo


def _matches(rows):
    frame = pd.DataFrame(
        rows,
        columns=[
            "as_of_date",
            "season",
            "competition_code",
            "home_team_key",
            "away_team_key",
            "result",
        ],
    )
    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"])
    return frame


D0 = date(2024, 8, 17)


def test_one_home_win_moves_both_ratings_by_the_same_amount():
    frame = _matches([(D0, "2024/25", "PL", 1, 2, "H")])
    settings = EloSettings(k=20, home_advantage=60)
    before = pre_match_elo(frame, settings)
    assert before.loc[0].tolist() == [1500.0, 1500.0]

    frame2 = pd.concat(
        [frame, _matches([(D0 + timedelta(days=7), "2024/25", "PL", 1, 2, "D")])],
        ignore_index=True,
    )
    after = pre_match_elo(frame2, settings)
    expected_home = 1 / (1 + 10 ** (-60 / 400))
    gain = 20 * (1 - expected_home)
    assert after.loc[1, "home_elo"] == pytest.approx(1500 + gain)
    assert after.loc[1, "away_elo"] == pytest.approx(1500 - gain)


def _league(seed: int, days: int = 30) -> pd.DataFrame:
    """Six clubs, three matches a day, random results."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(days):
        clubs = rng.permutation(6) + 1
        for i in range(0, 6, 2):
            rows.append(
                (
                    D0 + timedelta(days=3 * d),
                    "2024/25",
                    "PL",
                    clubs[i],
                    clubs[i + 1],
                    rng.choice(list("HDA")),
                )
            )
    return _matches(rows)


def test_a_rating_does_not_depend_on_its_own_day_or_any_later_one():
    frame = _league(seed=1)
    cutoff = frame["as_of_date"].iloc[len(frame) // 2]
    changed = frame.copy()
    flip = {"H": "A", "A": "D", "D": "H"}
    later = changed["as_of_date"] >= cutoff
    changed.loc[later, "result"] = changed.loc[later, "result"].map(flip)

    original, perturbed = pre_match_elo(frame), pre_match_elo(changed)

    # Every match up to and including the cutoff date has identical pre-match ratings,
    # although results on the cutoff date itself changed (L4: same-day results unseen).
    upto = frame["as_of_date"] <= cutoff
    pd.testing.assert_frame_equal(original[upto], perturbed[upto])
    assert not original[~upto].equals(perturbed[~upto])


def test_ratings_are_returned_in_the_frames_own_order():
    frame = _league(seed=2)
    shuffled = frame.sample(frac=1, random_state=0)
    pd.testing.assert_frame_equal(pre_match_elo(frame).loc[shuffled.index], pre_match_elo(shuffled))


def test_a_promoted_club_starts_at_the_bottom_three_of_last_season():
    season1 = _league(seed=3)
    final = pre_match_elo(
        pd.concat(
            [season1, _matches([(D0 + timedelta(days=300), "2025/26", "PL", 1, 2, "D")])],
            ignore_index=True,
        )
    ).iloc[-1]
    # Recompute every club's rating at the end of season 1 by adding a match per club.
    end = pd.concat(
        [
            season1,
            _matches(
                [(D0 + timedelta(days=300), "2025/26", "PL", c, c + 1, "D") for c in (1, 3, 5)]
            ),
        ],
        ignore_index=True,
    )
    ratings = pre_match_elo(end).iloc[-3:]
    end_of_season = np.concatenate([ratings["home_elo"], ratings["away_elo"]])
    bottom_three = np.sort(end_of_season)[:3].mean()

    promoted = _matches([(D0 + timedelta(days=300), "2025/26", "PL", 99, 1, "D")])
    with_promoted = pre_match_elo(pd.concat([season1, promoted], ignore_index=True)).iloc[-1]
    assert with_promoted["home_elo"] == pytest.approx(bottom_three)
    assert final["home_elo"] != pytest.approx(1500.0)


def test_in_the_first_season_everyone_starts_level():
    frame = _league(seed=4, days=1)
    assert set(pre_match_elo(frame).to_numpy().ravel()) == {1500.0}


def _league_with_strengths(seed: int, days: int = 300) -> pd.DataFrame:
    """Results driven by fixed club strengths, so ratings carry real signal."""
    rng = np.random.default_rng(seed)
    strength = np.linspace(-1.0, 1.0, 6)
    rows = []
    for d in range(days):
        clubs = rng.permutation(6)
        for i in range(0, 6, 2):
            h, a = clubs[i], clubs[i + 1]
            edge = strength[h] - strength[a] + 0.25 + rng.logistic()
            result = "H" if edge > 0.6 else "A" if edge < -0.6 else "D"
            rows.append((D0 + timedelta(days=3 * d), "2024/25", "PL", h + 1, a + 1, result))
    return _matches(rows)


def test_the_elo_forecast_backs_the_stronger_side_and_sums_to_one():
    # With random results a fitted slope can take either sign; with results driven by
    # strength it must be positive — which also pins the H/D/A column order, since a
    # reversed order would make P(home win) *fall* as the home edge grows.
    frame = _league_with_strengths(seed=5)
    ratings = pre_match_elo(frame)
    y = ev.outcome_index(frame["result"])
    model = EloForecast().fit(ratings, y)

    probe = pd.DataFrame({"home_elo": [1400.0, 1500.0, 1600.0], "away_elo": [1500.0] * 3})
    probs = model.predict_proba(probe)
    assert probs.sum(axis=1) == pytest.approx([1, 1, 1])
    assert list(probs[:, 0]) == sorted(probs[:, 0])  # P(home win) rises with the edge
    assert list(probs[:, 2]) == sorted(probs[:, 2], reverse=True)
    assert probs[1, 0] > probs[1, 2]  # level ratings: home advantage favours H


def test_base_rates_are_per_league_with_an_overall_fallback():
    frame = pd.DataFrame({"competition_code": ["PL", "PL", "PL", "PL", "SA", "SA"]})
    y = ev.outcome_index(["H", "H", "D", "A", "D", "D"])
    model = BaseRates().fit(frame, y)
    probs = model.predict_proba(pd.DataFrame({"competition_code": ["PL", "SA", "BL1"]}))
    assert probs[0] == pytest.approx([0.5, 0.25, 0.25])
    assert probs[1] == pytest.approx([0.0, 1.0, 0.0])
    assert probs[2] == pytest.approx([2 / 6, 3 / 6, 1 / 6])
