"""Model #2: goals per side, Dixon-Coles, turned into H/D/A (CLAUDE.md §8 #2).

Question: does modelling **goals** — each side's scoring rate — beat modelling the
**result** directly? And does fitting the rates to **xG** instead of goals help?

Each club has an attack ``a`` and a defence ``d``; in a match the home side's goals
are Poisson with mean ``λ = exp(μ + home + a[home] − d[away])`` and the away side's
with ``ν = exp(μ + a[away] − d[home])``. Dixon and Coles (1997) add a correction
``τ`` to the four low scorelines — 0-0, 1-0, 0-1, 1-1 — governed by one parameter
``ρ``, because independent Poissons get those wrong; and down-weight old matches by
``exp(−ξ · age in days)``. The score grid then sums to P(H), P(D), P(A).

Choices, and why:

- **One fit per league.** Clubs only meet within their league, so a joint fit could
  not tell one league's strengths from another's: shifting every attack and defence
  of a league by the same amount changes no prediction. Each league gets its own
  ``μ``, home advantage and ``ρ``.
- **Two targets.** ``goals`` fits everything jointly on scorelines. ``xg`` fits the
  rates to each side's xG with the same Poisson likelihood — a quasi-likelihood, since
  xG is not a count, which estimates the rates just as well — and then fits ``ρ`` alone
  on the actual scorelines with the rates held fixed, because ``ρ`` describes how
  *goals* cluster at low scores and has no meaning for xG. A match without xG is left
  out of the xG fit (Part 2 trap 7), never counted as zero.
- **A ridge penalty** on attacks and defences, so a club with few matches is pulled
  towards the league average instead of being fitted to three results.
- **Cold start (L7).** A club with no training match — promoted — starts at the mean
  attack and defence of the three weakest clubs (by ``a + d``: a higher ``d`` is a
  *better* defence, so overall strength adds the two) of its league's
  previous season in the training data: the clubs it most likely replaced, known
  before its first match. The same rule as the Elo baseline. In the first loaded
  season there is no previous season, and it starts at the league average.

The fit is point-in-time like every model here: it sees only the training fold, and
its decay is measured from the day after the fold's last training match.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, stats

MAX_GOALS = 10
RHO_BOUNDS = (-0.2, 0.2)
PROMOTED_FROM_BOTTOM = 3


@dataclass(frozen=True)
class DixonColesSettings:
    target: str  # "goals" or "xg"
    xi: float  # time decay per day; 0 weighs every match equally
    penalty: float  # ridge penalty on attacks and defences

    @property
    def name(self) -> str:
        return f"dixon_coles(target={self.target}, xi={self.xi:g}, penalty={self.penalty:g})"


def _tau_terms(x, y, lam, nu, rho):
    """log τ and its derivatives w.r.t. log λ, log ν and ρ, for every match.

    Only the four low scorelines have τ ≠ 1; everything else contributes zero.
    """
    tau = np.ones_like(lam)
    d_loglam = np.zeros_like(lam)
    d_lognu = np.zeros_like(lam)
    d_rho = np.zeros_like(lam)

    s00 = (x == 0) & (y == 0)
    tau[s00] = 1 - lam[s00] * nu[s00] * rho
    d_loglam[s00] = d_lognu[s00] = -lam[s00] * nu[s00] * rho / tau[s00]
    d_rho[s00] = -lam[s00] * nu[s00] / tau[s00]

    s01 = (x == 0) & (y == 1)
    tau[s01] = 1 + lam[s01] * rho
    d_loglam[s01] = lam[s01] * rho / tau[s01]
    d_rho[s01] = lam[s01] / tau[s01]

    s10 = (x == 1) & (y == 0)
    tau[s10] = 1 + nu[s10] * rho
    d_lognu[s10] = nu[s10] * rho / tau[s10]
    d_rho[s10] = nu[s10] / tau[s10]

    s11 = (x == 1) & (y == 1)
    tau[s11] = 1 - rho
    d_rho[s11] = -1 / tau[s11]

    return np.log(np.clip(tau, 1e-12, None)), d_loglam, d_lognu, d_rho


class _League:
    """The fitted parameters of one league."""

    def __init__(self, teams, mu, home, rho, attack, defence):
        self.index = {t: i for i, t in enumerate(teams)}
        self.mu, self.home, self.rho = mu, home, rho
        self.attack, self.defence = attack, defence

    def strengths(self, team, prior):
        i = self.index.get(team)
        return prior if i is None else (self.attack[i], self.defence[i])

    def strength(self, team) -> float:
        """Overall strength: attack **plus** defence.

        ``d`` enters the opponent's rate as ``− d``, so a higher ``d`` concedes less —
        a strong side has a high ``a`` *and* a high ``d``. (Ranking by ``a − d`` was
        this module's first version, and it called strong defences weak.)
        """
        i = self.index[team]
        return float(self.attack[i] + self.defence[i])


def _rates(params, n, h, a):
    mu, home = params[0], params[1]
    att, dfc = params[3 : 3 + n], params[3 + n : 3 + 2 * n]
    log_lam = mu + home + att[h] - dfc[a]
    log_nu = mu + att[a] - dfc[h]
    return log_lam, log_nu


def _objective(params, n, h, a, x, y, w, penalty, with_tau, goals_x, goals_y):
    """Weighted negative log-likelihood + ridge penalty, and its gradient.

    ``params`` is ``[μ, home, ρ, attack × n, defence × n]``. With ``with_tau`` the
    Dixon-Coles correction is applied to the integer scorelines ``goals_x/y``.
    """
    log_lam, log_nu = _rates(params, n, h, a)
    lam, nu = np.exp(log_lam), np.exp(log_nu)
    ll = x * log_lam - lam + y * log_nu - nu
    g_lam = x - lam
    g_nu = y - nu
    g_rho = 0.0
    if with_tau:
        log_tau, dl, dn, dr = _tau_terms(goals_x, goals_y, lam, nu, params[2])
        ll = ll + log_tau
        g_lam = g_lam + dl
        g_nu = g_nu + dn
        g_rho = float((w * dr).sum())
    g_lam, g_nu = w * g_lam, w * g_nu

    att, dfc = params[3 : 3 + n], params[3 + n : 3 + 2 * n]
    value = -(w * ll).sum() + penalty * ((att**2).sum() + (dfc**2).sum())

    grad = np.zeros_like(params)
    grad[0] = -(g_lam.sum() + g_nu.sum())
    grad[1] = -g_lam.sum()
    grad[2] = -g_rho
    grad[3 : 3 + n] = -(np.bincount(h, g_lam, n) + np.bincount(a, g_nu, n)) + 2 * penalty * att
    grad[3 + n :] = np.bincount(a, g_lam, n) + np.bincount(h, g_nu, n) + 2 * penalty * dfc
    return value, grad


def _fit_league(rows: pd.DataFrame, settings: DixonColesSettings, reference) -> _League:
    teams = sorted(set(rows["home_team_key"]) | set(rows["away_team_key"]))
    index = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    h_all = rows["home_team_key"].map(index).to_numpy()
    a_all = rows["away_team_key"].map(index).to_numpy()
    gx = rows["target_home_goals"].to_numpy(dtype=float)
    gy = rows["target_away_goals"].to_numpy(dtype=float)
    age = (reference - pd.to_datetime(rows["as_of_date"])).dt.days.to_numpy()
    w_all = np.exp(-settings.xi * age)
    w_all = w_all / w_all.mean()

    if settings.target == "goals":
        keep = np.ones(len(rows), dtype=bool)
        x, y = gx, gy
    elif settings.target == "xg":
        hx = rows["target_home_xg"].astype("float64").to_numpy()
        ay = rows["target_away_xg"].astype("float64").to_numpy()
        keep = ~(np.isnan(hx) | np.isnan(ay))  # unpaired matches: missing, not zero
        x, y = hx, ay
    else:
        raise ValueError(f"unknown target {settings.target!r}")

    with_tau = settings.target == "goals"
    start = np.zeros(3 + 2 * n)
    start[0] = np.log(max(np.nanmean(np.concatenate([x[keep], y[keep]])), 0.1))
    start[1] = 0.2
    bounds = [(None, None), (None, None), RHO_BOUNDS if with_tau else (0.0, 0.0)] + [
        (None, None)
    ] * (2 * n)
    result = optimize.minimize(
        _objective,
        start,
        args=(
            n,
            h_all[keep],
            a_all[keep],
            x[keep],
            y[keep],
            w_all[keep],
            settings.penalty,
            with_tau,
            gx[keep],
            gy[keep],
        ),
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
    )
    params = result.x

    if not with_tau:
        # ρ from the actual scorelines, every match (xG or not), rates held fixed.
        log_lam, log_nu = _rates(params, n, h_all, a_all)
        lam, nu = np.exp(log_lam), np.exp(log_nu)

        def neg_ll(rho):
            return -(w_all * _tau_terms(gx, gy, lam, nu, rho)[0]).sum()

        params[2] = optimize.minimize_scalar(neg_ll, bounds=RHO_BOUNDS, method="bounded").x

    return _League(
        teams, params[0], params[1], params[2], params[3 : 3 + n], params[3 + n : 3 + 2 * n]
    )


def _previous_season(season: str) -> str:
    start = int(season[:4]) - 1
    return f"{start}/{(start + 1) % 100:02d}"


class DixonColes:
    """Fit on a training fold; predict H/D/A (and expected goals) for later matches."""

    def __init__(self, settings: DixonColesSettings):
        self.settings = settings

    def fit(self, frame: pd.DataFrame) -> DixonColes:
        reference = pd.to_datetime(frame["as_of_date"]).max() + pd.Timedelta(days=1)
        self.leagues_: dict[str, _League] = {}
        self.clubs_by_season_: dict[tuple[str, str], set] = {}
        for league, rows in frame.groupby("competition_code"):
            self.leagues_[league] = _fit_league(rows, self.settings, reference)
        for (league, season), rows in frame.groupby(["competition_code", "season"]):
            self.clubs_by_season_[(league, season)] = set(rows["home_team_key"]) | set(
                rows["away_team_key"]
            )
        return self

    def _prior(self, league: str, season: str) -> tuple[float, float]:
        fitted = self.leagues_[league]
        last = self.clubs_by_season_.get((league, _previous_season(season)), set())
        known = [c for c in last if c in fitted.index]
        if not known:
            return 0.0, 0.0
        weakest = sorted(known, key=fitted.strength)
        chosen = [fitted.strengths(c, None) for c in weakest[:PROMOTED_FROM_BOTTOM]]
        return float(np.mean([s[0] for s in chosen])), float(np.mean([s[1] for s in chosen]))

    def expected_goals(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """λ (home), ν (away) and ρ for every match."""
        lam, nu, rho = (np.empty(len(frame)) for _ in range(3))
        for i, m in enumerate(frame.itertuples(index=False)):
            league = self.leagues_.get(m.competition_code)
            if league is None:
                raise ValueError(f"no training matches for league {m.competition_code!r}")
            prior = self._prior(m.competition_code, m.season)
            ah, dh = league.strengths(m.home_team_key, prior)
            aa, da = league.strengths(m.away_team_key, prior)
            lam[i] = np.exp(league.mu + league.home + ah - da)
            nu[i] = np.exp(league.mu + aa - dh)
            rho[i] = league.rho
        return lam, nu, rho

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        lam, nu, rho = self.expected_goals(frame)
        return score_grid_to_hda(lam, nu, rho)


def score_grid_to_hda(lam, nu, rho) -> np.ndarray:
    """P(H), P(D), P(A) from the Dixon-Coles score grid, 0–MAX_GOALS each side."""
    goals = np.arange(MAX_GOALS + 1)
    home = stats.poisson.pmf(goals[None, :], np.asarray(lam)[:, None])
    away = stats.poisson.pmf(goals[None, :], np.asarray(nu)[:, None])
    grid = home[:, :, None] * away[:, None, :]
    grid[:, 0, 0] *= 1 - lam * nu * rho
    grid[:, 0, 1] *= 1 + lam * rho
    grid[:, 1, 0] *= 1 + nu * rho
    grid[:, 1, 1] *= 1 - rho
    grid = np.clip(grid, 0, None)
    grid /= grid.sum(axis=(1, 2), keepdims=True)
    return np.stack(
        [
            np.tril(grid, -1).sum(axis=(1, 2)),  # home goals > away goals
            np.trace(grid, axis1=1, axis2=2),
            np.triu(grid, 1).sum(axis=(1, 2)),
        ],
        axis=1,
    )
