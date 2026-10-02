"""Walk-forward splits never let a fold see its own future (CLAUDE.md §5 L5, L6)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from football_ml import splits


def _round_robin(season: str, competition: str, clubs: list[int], start: date, *, played=None):
    """A double round robin, one round a week; ``played`` truncates it (in progress).

    Each round's matches share one date, so same-day handling is always exercised.
    """
    n = len(clubs)
    rotation = clubs[:]
    rounds = []
    for _ in range(n - 1):
        rounds.append([(rotation[i], rotation[n - 1 - i]) for i in range(n // 2)])
        rotation = [rotation[0], rotation[-1], *rotation[1:-1]]
    rounds += [[(away, home) for home, away in r] for r in rounds]
    rows = [
        {
            "as_of_date": start + timedelta(weeks=week),
            "season": season,
            "competition_code": competition,
            "home_team_key": home,
            "away_team_key": away,
        }
        for week, fixtures in enumerate(rounds)
        for home, away in fixtures
    ]
    return rows[:played] if played is not None else rows


@pytest.fixture
def frame() -> pd.DataFrame:
    """Two complete seasons of a 20-club and an 18-club league, and a season begun."""
    rows = []
    for season, start in [("2024/25", date(2024, 8, 16)), ("2025/26", date(2025, 8, 15))]:
        rows += _round_robin(season, "PL", list(range(1, 21)), start)
        rows += _round_robin(season, "BL1", list(range(101, 119)), start)
    rows += _round_robin("2026/27", "PL", list(range(1, 21)), date(2026, 8, 14), played=30)
    return pd.DataFrame(rows).sort_values("as_of_date", kind="stable").reset_index(drop=True)


def _dates(frame, idx):
    return pd.to_datetime(frame["as_of_date"].iloc[idx]).dt.date


def test_complete_seasons_are_read_from_the_data(frame):
    assert splits.complete_seasons(frame) == ["2024/25", "2025/26"]


def test_an_18_club_league_counts_as_complete_without_a_special_case(frame):
    bl1 = frame[frame["competition_code"] == "BL1"]
    assert len(bl1[bl1["season"] == "2024/25"]) == 18 * 17
    assert "2024/25" in splits.complete_seasons(bl1)


def test_the_holdout_is_the_latest_complete_season_and_live_is_excluded(frame):
    plan = splits.plan(frame)

    assert plan.holdout_season == "2025/26"
    assert plan.live_seasons == ["2026/27"]

    every_test = np.concatenate([f.test for f in plan.validation + plan.holdout])
    every_train = np.concatenate([f.train for f in plan.validation + plan.holdout])
    live = np.flatnonzero(frame["season"] == "2026/27")
    assert not np.isin(live, every_test).any()
    assert not np.isin(live, every_train).any()


@pytest.mark.parametrize("period", ["validation", "holdout"])
def test_every_training_match_is_strictly_before_every_test_match(frame, period):
    for fold in getattr(splits.plan(frame), period):
        assert _dates(frame, fold.train).max() < _dates(frame, fold.test).min(), fold.label
        assert _dates(frame, fold.train).max() < fold.test_start


def test_no_date_is_split_between_train_and_test(frame):
    for fold in splits.plan(frame).validation + splits.plan(frame).holdout:
        assert not set(_dates(frame, fold.train)) & set(_dates(frame, fold.test))


def test_test_blocks_do_not_overlap_and_cover_their_period(frame):
    plan = splits.plan(frame)
    for folds, season_mask in [
        (plan.holdout, frame["season"] == "2025/26"),
    ]:
        tests = np.concatenate([f.test for f in folds])
        assert len(tests) == len(set(tests))
        assert set(tests) == set(np.flatnonzero(season_mask))


def test_validation_never_reaches_the_holdout(frame):
    plan = splits.plan(frame)
    for fold in plan.validation:
        assert fold.test_end <= plan.holdout_start
        assert _dates(frame, fold.test).max() < plan.holdout_start


def test_the_burn_in_is_not_a_validation_fold(frame):
    plan = splits.plan(frame, burn_in_months=3)
    assert plan.validation[0].test_start == date(2024, 11, 1)
    assert len(plan.validation[0].train) > 0


def test_training_expands_fold_by_fold(frame):
    folds = splits.plan(frame).validation + splits.plan(frame).holdout
    sizes = [len(f.train) for f in folds]
    assert sizes == sorted(sizes)
    for earlier, later in zip(folds, folds[1:], strict=False):
        assert set(earlier.train) <= set(later.train)


def test_no_complete_season_is_refused(frame):
    with pytest.raises(ValueError, match="No complete season"):
        splits.plan(frame[frame["season"] == "2026/27"])


def test_aligned_folds_keep_the_months_and_never_split_a_match(frame):
    plan = splits.plan(frame)
    # Ten "shots" per match, each carrying its match's date and id.
    shots = frame.loc[frame.index.repeat(10), ["as_of_date"]].reset_index(names="match")
    shot_plan = splits.aligned(plan, shots["as_of_date"])

    assert [f.label for f in shot_plan.validation] == [f.label for f in plan.validation]
    assert [f.label for f in shot_plan.holdout] == [f.label for f in plan.holdout]
    for fold in shot_plan.validation + shot_plan.holdout:
        train_matches = set(shots.loc[fold.train, "match"])
        test_matches = set(shots.loc[fold.test, "match"])
        assert not train_matches & test_matches
        assert _dates(shots, fold.train).max() < fold.test_start
    live = pd.to_datetime(shots["as_of_date"]) >= pd.Timestamp(plan.holdout[-1].test_end)
    every = np.concatenate([f.test for f in shot_plan.validation + shot_plan.holdout])
    assert not np.isin(np.flatnonzero(live), every).any()
