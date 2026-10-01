"""Dixon-Coles: the gradient is right, the fit recovers known strengths, and the
score grid turns into H/D/A correctly."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from scipy import optimize, stats

from football_ml.models.dixon_coles import (
    DixonColes,
    DixonColesSettings,
    _objective,
    score_grid_to_hda,
)

GOALS = DixonColesSettings(target="goals", xi=0.0, penalty=0.1)
XG = DixonColesSettings(target="xg", xi=0.0, penalty=0.1)


@pytest.mark.parametrize("with_tau", [True, False])
def test_the_analytic_gradient_matches_a_numerical_one(with_tau):
    rng = np.random.default_rng(0)
    n, m = 6, 200
    h, a = rng.integers(0, n, m), rng.integers(0, n, m)
    gx, gy = rng.poisson(1.4, m).astype(float), rng.poisson(1.1, m).astype(float)
    x, y = (gx, gy) if with_tau else (rng.gamma(2, 0.7, m), rng.gamma(2, 0.55, m))
    w = rng.uniform(0.5, 1.5, m)
    params = np.concatenate([[0.2, 0.25, -0.08 if with_tau else 0.0], rng.normal(0, 0.3, 2 * n)])
    args = (n, h, a, x, y, w, 0.7, with_tau, gx, gy)

    error = optimize.check_grad(
        lambda p: _objective(p, *args)[0], lambda p: _objective(p, *args)[1], params
    )
    assert error < 1e-4 * np.linalg.norm(_objective(params, *args)[1])


def test_the_score_grid_sums_to_one_and_is_symmetric_for_equal_sides():
    probs = score_grid_to_hda(np.array([1.3, 1.3, 2.5]), np.array([1.3, 0.6, 0.4]), np.zeros(3))
    assert probs.sum(axis=1) == pytest.approx(np.ones(3))
    assert probs[0, 0] == pytest.approx(probs[0, 2])  # equal rates: P(H) = P(A)
    assert probs[2, 0] > probs[1, 0] > probs[0, 0]


def test_without_rho_the_draw_is_the_independent_poisson_draw():
    lam, nu = 1.4, 1.1
    goals = np.arange(40)
    expected = (stats.poisson.pmf(goals, lam) * stats.poisson.pmf(goals, nu)).sum()
    assert score_grid_to_hda(np.array([lam]), np.array([nu]), np.array([0.0]))[0, 1] == (
        pytest.approx(expected, abs=1e-6)
    )


def test_a_negative_rho_makes_low_score_draws_more_likely():
    base = score_grid_to_hda(np.array([1.2]), np.array([1.0]), np.array([0.0]))[0, 1]
    corrected = score_grid_to_hda(np.array([1.2]), np.array([1.0]), np.array([-0.1]))[0, 1]
    assert corrected > base


def _season(
    strength: dict[int, tuple[float, float]],
    seed: int,
    *,
    rounds: int = 6,
    season: str = "2024/25",
    start: date = date(2024, 8, 17),
) -> pd.DataFrame:
    """Every pair meets `rounds` times at each venue; goals drawn from the true model."""
    rng = np.random.default_rng(seed)
    clubs = sorted(strength)
    rows, day = [], start
    for _ in range(rounds):
        for home in clubs:
            for away in clubs:
                if home == away:
                    continue
                lam = np.exp(0.1 + 0.25 + strength[home][0] - strength[away][1])
                nu = np.exp(0.1 + strength[away][0] - strength[home][1])
                rows.append(
                    {
                        "as_of_date": pd.Timestamp(day),
                        "season": season,
                        "competition_code": "PL",
                        "home_team_key": home,
                        "away_team_key": away,
                        "target_home_goals": rng.poisson(lam),
                        "target_away_goals": rng.poisson(nu),
                        "target_home_xg": lam * rng.uniform(0.8, 1.2),
                        "target_away_xg": nu * rng.uniform(0.8, 1.2),
                    }
                )
                day += timedelta(days=1)
    return pd.DataFrame(rows)


STRENGTH = {1: (0.5, 0.3), 2: (0.2, 0.1), 3: (0.0, 0.0), 4: (-0.3, -0.2), 5: (-0.4, -0.2)}


@pytest.mark.parametrize("settings", [GOALS, XG], ids=["goals", "xg"])
def test_the_fit_recovers_the_order_of_club_strengths(settings):
    model = DixonColes(settings).fit(_season(STRENGTH, seed=1))
    league = model.leagues_["PL"]
    # Strength is a + d (a high d concedes less). True: 0.8, 0.3, 0, −0.5, −0.6.
    assert sorted(league.index, key=league.strength) == [5, 4, 3, 2, 1]
    assert 0.1 < league.home < 0.45  # true home advantage 0.25


def test_the_strong_home_side_is_favoured():
    model = DixonColes(GOALS).fit(_season(STRENGTH, seed=2))
    probe = pd.DataFrame(
        {"competition_code": ["PL", "PL"], "season": ["2024/25"] * 2,
         "home_team_key": [1, 5], "away_team_key": [5, 1]}
    )  # fmt: skip
    probs = model.predict_proba(probe)
    assert probs[0, 0] > 0.6 > probs[1, 0]
    assert probs.sum(axis=1) == pytest.approx([1, 1])


def test_matches_without_xg_are_left_out_of_the_xg_fit_not_counted_as_zero():
    season = _season(STRENGTH, seed=3)
    with_gaps = season.copy()
    gaps = with_gaps.index[::7]
    with_gaps.loc[gaps, ["target_home_xg", "target_away_xg"]] = np.nan

    fitted_gaps = DixonColes(XG).fit(with_gaps).leagues_["PL"]
    zeroed = with_gaps.fillna({"target_home_xg": 0.0, "target_away_xg": 0.0})
    fitted_zero = DixonColes(XG).fit(zeroed).leagues_["PL"]
    fitted_dropped = DixonColes(XG).fit(season.drop(index=gaps)).leagues_["PL"]

    # Leaving the gaps out is the same as never having the rows for the rate fit...
    assert fitted_gaps.mu == pytest.approx(fitted_dropped.mu, abs=1e-6)
    np.testing.assert_allclose(fitted_gaps.attack, fitted_dropped.attack, atol=1e-5)
    # ...and treating them as zero would drag the scoring rate down.
    assert fitted_zero.mu < fitted_gaps.mu - 0.05


# Club 2 defends well and attacks badly: strong overall (a + d = +0.4), but ranked
# the weakest club of all by a − d (−0.8). That is the mistake this test pins.
DEFENSIVE = {1: (0.5, 0.3), 2: (-0.2, 0.6), 3: (0.0, 0.0), 4: (-0.3, -0.2), 5: (-0.4, -0.2)}


def test_a_promoted_club_starts_at_the_weakest_three_of_last_season():
    model = DixonColes(GOALS).fit(_season(DEFENSIVE, seed=4, rounds=20))
    league = model.leagues_["PL"]
    assert sorted(league.index, key=league.strength)[:3] == [5, 4, 3]  # not club 2

    weakest = [league.index[c] for c in (3, 4, 5)]
    prior_attack = league.attack[weakest].mean()
    prior_defence = league.defence[weakest].mean()

    probe = pd.DataFrame(
        {"competition_code": ["PL"], "season": ["2025/26"], "home_team_key": [99],
         "away_team_key": [1]}
    )  # fmt: skip
    lam, nu, _ = model.expected_goals(probe)
    i1 = league.index[1]
    assert lam[0] == pytest.approx(
        np.exp(league.mu + league.home + prior_attack - league.defence[i1])
    )
    assert nu[0] == pytest.approx(np.exp(league.mu + league.attack[i1] - prior_defence))


def test_in_the_first_loaded_season_a_new_club_starts_at_the_league_average():
    model = DixonColes(GOALS).fit(_season(STRENGTH, seed=7))
    league = model.leagues_["PL"]
    probe = pd.DataFrame(
        {"competition_code": ["PL"], "season": ["2024/25"], "home_team_key": [99],
         "away_team_key": [3]}
    )  # fmt: skip
    lam, _, _ = model.expected_goals(probe)
    i3 = league.index[3]
    assert lam[0] == pytest.approx(np.exp(league.mu + league.home + 0.0 - league.defence[i3]))


def test_time_decay_weights_recent_form_more():
    early = _season({1: (0.6, 0.3), 2: (0.0, 0.0), 3: (-0.3, -0.2)}, seed=5, rounds=6)
    late = _season({1: (-0.6, -0.3), 2: (0.0, 0.0), 3: (0.3, 0.2)}, seed=6, rounds=6,
                   start=date(2025, 3, 1))  # fmt: skip
    both = pd.concat([early, late], ignore_index=True)
    flat = DixonColes(DixonColesSettings("goals", 0.0, 0.1)).fit(both).leagues_["PL"]
    decayed = DixonColes(DixonColesSettings("goals", 0.02, 0.1)).fit(both).leagues_["PL"]
    club1 = lambda lg: lg.attack[lg.index[1]] - lg.defence[lg.index[1]]  # noqa: E731
    assert club1(decayed) < club1(flat)  # club 1 declined; decay sees it sooner
