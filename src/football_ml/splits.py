"""Walk-forward splits — the only place a train/test split is made (CLAUDE.md §7).

Every split is forward in time (L5): a fold trains on every match **strictly
before** a date and predicts the calendar month that starts on it, then the next
fold moves a month on and trains on everything up to there. Nothing is shuffled,
and matches on one date are never split between train and test.

The data is cut into three periods, in order:

``validation``
    Every month from the end of a burn-in to the start of the hold-out. Hyper-
    parameters and model choice are fixed here, and only here (L6). The burn-in
    exists because the first loaded months have almost no history — every club is
    cold (L7) — and a fold trained on two matchdays measures nothing.

``holdout``
    The most recent **complete** season, walked forward month by month with
    settings frozen. Scored once.

``live``
    Anything after the hold-out — the season in progress. In no fold at all: it is
    the prospective test (CLAUDE.md §7, phase 5), and looking at it now would spend
    it.

Which season is complete is read from the data, not from a list: a competition's
season is complete when it has played every fixture of a double round robin,
``clubs × (clubs − 1)`` matches — Part 2's rule, so 18-club leagues need no special
case and the in-progress season joins by itself when it finishes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

BURN_IN_MONTHS = 3


@dataclass(frozen=True)
class Fold:
    """One step of a walk-forward: train on ``train``, predict ``test``.

    ``train`` and ``test`` are positional indices into the frame the plan was made
    from. Every training match is strictly before ``test_start``; every test match is
    in ``[test_start, test_end)``.
    """

    label: str
    test_start: date
    test_end: date
    train: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class Plan:
    validation: list[Fold]
    holdout: list[Fold]
    holdout_season: str
    live_seasons: list[str]

    @property
    def holdout_start(self) -> date:
        return self.holdout[0].test_start


def complete_seasons(frame: pd.DataFrame) -> list[str]:
    """Seasons in which every competition played its full double round robin."""
    clubs = (
        pd.concat(
            [
                frame[["season", "competition_code", "home_team_key"]].set_axis(
                    ["season", "competition_code", "team_key"], axis=1
                ),
                frame[["season", "competition_code", "away_team_key"]].set_axis(
                    ["season", "competition_code", "team_key"], axis=1
                ),
            ]
        )
        .groupby(["season", "competition_code"])["team_key"]
        .nunique()
    )
    matches = frame.groupby(["season", "competition_code"]).size()
    complete = matches == clubs * (clubs - 1)
    by_season = complete.groupby(level="season").all()
    return sorted(by_season[by_season].index)


def _month_starts(first: date, last: date) -> list[date]:
    """First day of every month from ``first``'s month through ``last``'s month."""
    months = pd.period_range(pd.Period(first, "M"), pd.Period(last, "M"), freq="M")
    return [m.start_time.date() for m in months]


def monthly_folds(dates: pd.Series, start: date, end: date) -> list[Fold]:
    """One fold per calendar month with matches in ``[start, end)``.

    Training is expanding: everything strictly before the month. Months without a
    match — the summer break — produce no fold.
    """
    as_dates = pd.to_datetime(dates).dt.date.to_numpy()
    folds = []
    for month_start in _month_starts(start, end):
        block_start = max(month_start, start)
        block_end = min((pd.Timestamp(month_start) + pd.offsets.MonthBegin(1)).date(), end)
        test = np.flatnonzero((as_dates >= block_start) & (as_dates < block_end))
        if test.size == 0:
            continue
        train = np.flatnonzero(as_dates < block_start)
        folds.append(
            Fold(
                label=month_start.strftime("%Y-%m"),
                test_start=block_start,
                test_end=block_end,
                train=train,
                test=test,
            )
        )
    return folds


def plan(frame: pd.DataFrame, *, burn_in_months: int = BURN_IN_MONTHS) -> Plan:
    """Cut ``frame`` (one row per match) into validation, hold-out and live periods.

    Needs ``as_of_date``, ``season``, ``competition_code``, ``home_team_key`` and
    ``away_team_key``. The hold-out is the latest complete season; validation is
    every month between the burn-in and the hold-out's first match.
    """
    complete = complete_seasons(frame)
    if not complete:
        raise ValueError("No complete season in the data — nothing to hold out.")
    holdout_season = complete[-1]

    dates = pd.to_datetime(frame["as_of_date"]).dt.date
    in_holdout = frame["season"] == holdout_season
    holdout_start = dates[in_holdout].min()
    holdout_end = (pd.Timestamp(dates[in_holdout].max()) + pd.Timedelta(days=1)).date()

    first = dates.min()
    validation_start = (pd.Timestamp(first).to_period("M") + burn_in_months).start_time.date()
    if validation_start >= holdout_start:
        raise ValueError(
            f"No validation period: the burn-in ends {validation_start}, "
            f"but the hold-out season {holdout_season} starts {holdout_start}."
        )

    return Plan(
        validation=monthly_folds(dates, validation_start, holdout_start),
        holdout=monthly_folds(dates, holdout_start, holdout_end),
        holdout_season=holdout_season,
        live_seasons=sorted(set(frame.loc[dates >= holdout_end, "season"])),
    )


def aligned(plan: Plan, dates: pd.Series) -> Plan:
    """The same periods and month boundaries as ``plan``, for other rows — shots.

    Model #3 is scored on the same months as the match models, so its validation and
    hold-out mean the same thing. Every shot of a match shares its date, so a match is
    never split between training and test (L5); rows after the hold-out (the live
    season) fall in no fold.
    """
    return Plan(
        validation=monthly_folds(dates, plan.validation[0].test_start, plan.holdout_start),
        holdout=monthly_folds(dates, plan.holdout_start, plan.holdout[-1].test_end),
        holdout_season=plan.holdout_season,
        live_seasons=plan.live_seasons,
    )
