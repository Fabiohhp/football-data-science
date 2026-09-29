"""Metrics against values worked out by hand, and the bootstrap's basic honesty."""

from __future__ import annotations

import numpy as np
import pytest

from football_ml import evaluate as ev

P = [[0.5, 0.3, 0.2]]


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        # cumulative forecast (0.5, 0.8)
        (0, ((0.5 - 1) ** 2 + (0.8 - 1) ** 2) / 2),  # H: cum outcome (1, 1) -> 0.145
        (1, ((0.5 - 0) ** 2 + (0.8 - 1) ** 2) / 2),  # D: cum outcome (0, 1) -> 0.145
        (2, ((0.5 - 0) ** 2 + (0.8 - 0) ** 2) / 2),  # A: cum outcome (0, 0) -> 0.445
    ],
)
def test_rps_by_hand(outcome, expected):
    assert ev.rps(P, [outcome])[0] == pytest.approx(expected)
    assert expected == pytest.approx([0.145, 0.145, 0.445][outcome])


def test_rps_rewards_probability_near_the_outcome_where_brier_cannot_tell():
    # At a home win, 50/50 between H and D is closer than 50/50 between H and A.
    near, far = [[0.5, 0.5, 0.0]], [[0.5, 0.0, 0.5]]
    assert ev.rps(near, [0])[0] == pytest.approx(0.125)
    assert ev.rps(far, [0])[0] == pytest.approx(0.25)
    assert ev.brier(near, [0])[0] == pytest.approx(ev.brier(far, [0])[0]) == pytest.approx(0.5)
    assert ev.log_loss(near, [0])[0] == pytest.approx(ev.log_loss(far, [0])[0])


def test_a_certain_correct_forecast_scores_zero():
    probs = np.eye(3)
    y = [0, 1, 2]
    assert ev.rps(probs, y) == pytest.approx([0, 0, 0])
    assert ev.brier(probs, y) == pytest.approx([0, 0, 0])
    assert ev.log_loss(probs, y) == pytest.approx([0, 0, 0])


def test_log_loss_and_brier_by_hand():
    assert ev.log_loss(P, [1])[0] == pytest.approx(-np.log(0.3))
    # (0.5-1)^2 + 0.3^2 + 0.2^2 = 0.25 + 0.09 + 0.04
    assert ev.brier(P, [0])[0] == pytest.approx(0.38)


def test_log_loss_of_a_zero_probability_is_large_not_infinite():
    assert np.isfinite(ev.log_loss([[1.0, 0.0, 0.0]], [2])[0])


def test_outcomes_are_mapped_in_h_d_a_order():
    assert list(ev.outcome_index(["H", "D", "A", "H"])) == [0, 1, 2, 0]
    with pytest.raises(ValueError, match="unknown outcome"):
        ev.outcome_index(["X"])


def test_malformed_forecasts_are_refused():
    with pytest.raises(ValueError, match="sum to 1"):
        ev.rps([[0.5, 0.5, 0.5]], [0])
    with pytest.raises(ValueError, match=r"\(n, 3\)"):
        ev.rps([[0.5, 0.5]], [0])


def test_calibration_per_class():
    probs = np.array([[0.62, 0.25, 0.13]] * 10)
    y = np.array([0] * 6 + [1] * 3 + [2] * 1)
    rows = {(r["outcome"], r["bin_low"]): r for r in ev.calibration(probs, y)}
    assert rows[("H", 0.6)]["observed_rate"] == pytest.approx(0.6)
    assert rows[("H", 0.6)]["mean_predicted"] == pytest.approx(0.62)
    assert rows[("D", 0.2)]["observed_rate"] == pytest.approx(0.3)
    assert rows[("A", 0.1)]["n"] == 10


def test_a_probability_of_exactly_one_lands_in_the_top_bin():
    rows = ev.calibration([[1.0, 0.0, 0.0]], [0])
    assert [r["bin_low"] for r in rows if r["outcome"] == "H"] == [0.9]


def test_draw_summary_exposes_a_model_that_never_backs_a_draw():
    probs = np.array([[0.45, 0.30, 0.25]] * 4)
    summary = ev.draw_summary(probs, [0, 1, 1, 2])
    assert summary == {
        "actual_draw_rate": 0.5,
        "mean_predicted_draw": pytest.approx(0.30),
        "draw_most_likely_share": 0.0,
    }


def test_paired_difference_of_identical_forecasts_is_exactly_zero():
    losses = np.random.default_rng(1).random(200)
    d = ev.paired_difference(losses, losses)
    assert (d.value, d.low, d.high) == (0.0, 0.0, 0.0)
    assert d.verdict == "cannot be distinguished"


def test_a_constant_improvement_is_found_with_a_degenerate_interval():
    reference = np.random.default_rng(2).random(300)
    d = ev.paired_difference(
        reference - 0.01, reference, candidate="model", reference="elo", resamples=500
    )
    assert d.value == pytest.approx(-0.01)
    assert d.low == pytest.approx(-0.01) and d.high == pytest.approx(-0.01)
    assert d.verdict == "model is better"


def test_pairing_is_what_makes_a_small_consistent_gain_visible():
    # Losses vary a lot between matches but the candidate is always 0.002 lower:
    # two separate intervals would overlap completely; the paired one excludes zero.
    reference = np.random.default_rng(3).random(1000)
    d = ev.paired_difference(reference - 0.002, reference, resamples=1000)
    assert d.high < 0


def test_noise_is_reported_as_indistinguishable():
    # The same losses, assigned to matches in a different order: no difference on
    # average, a lot of match-to-match noise.
    a = np.random.default_rng(4).random(500)
    b = np.random.default_rng(5).permutation(a)
    d = ev.paired_difference(a, b, resamples=2000)
    assert d.value == pytest.approx(0.0)
    assert d.verdict == "cannot be distinguished"


def test_the_bootstrap_is_seeded():
    rng = np.random.default_rng(5)
    a, b = rng.random(100), rng.random(100)
    first = ev.paired_difference(a, b, resamples=500)
    assert first == ev.paired_difference(a, b, resamples=500)
    assert first != ev.paired_difference(a, b, resamples=500, seed=1)
