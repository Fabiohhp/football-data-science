"""Scoring probabilistic forecasts of H/D/A, and comparing two of them honestly.

The product of every model here is a probability for each outcome, so the metrics
score probabilities (CLAUDE.md §7) — accuracy never leads, because it throws them
away:

* :func:`rps` — **ranked probability score**, the headline. H, D and A are
  *ordered* (a draw is between the two wins), and RPS rewards putting probability
  near the outcome: at a home win, "50% H, 50% D" beats "50% H, 50% A", which
  log loss and Brier score identically. Not in scikit-learn; lower is better.
* :func:`log_loss` and :func:`brier` beside it.
* :func:`calibration` — per class, because a model can be well calibrated overall
  while never believing in draws (L11); :func:`draw_summary` says so directly.
* :func:`paired_difference` — model A against model B as a **paired bootstrap over
  matches** of the per-match score difference. Both forecasts score the same
  matches, so resampling matches keeps the pairing; comparing two separate
  intervals would overstate the uncertainty. Seeded, as Part 2's ``stats`` is, so
  the committed report reproduces.

Outcome order is always ``OUTCOMES = ("H", "D", "A")``: column 0 is the home win.

Shots (model #3) are binary — goal or not — and have their own small set:
:func:`binary_scores` (log loss and Brier, with AUC beside them for discrimination)
and :func:`binary_calibration`. Their comparisons pass ``groups`` to
:func:`paired_difference`, which then resamples **matches**, not shots: shots within a
match share a game state and a goalkeeper, so they are not independent (L5), and
resampling them one by one would make every interval too narrow.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

OUTCOMES = ("H", "D", "A")
DEFAULT_SEED = 20260920  # Part 2's stats.DEFAULT_SEED, so the portfolio shares one
DEFAULT_RESAMPLES = 10_000
DEFAULT_CONFIDENCE = 0.95
_EPS = 1e-15


def outcome_index(results) -> np.ndarray:
    """``["H", "D", "A", ...]`` → ``[0, 1, 2, ...]``."""
    lookup = {o: i for i, o in enumerate(OUTCOMES)}
    try:
        return np.array([lookup[r] for r in results], dtype=int)
    except KeyError as exc:
        raise ValueError(f"unknown outcome {exc.args[0]!r}; expected one of {OUTCOMES}") from exc


def _check(probs, y) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=int)
    if probs.ndim != 2 or probs.shape[1] != len(OUTCOMES):
        raise ValueError(f"probs must be (n, {len(OUTCOMES)}), got {probs.shape}")
    if len(probs) != len(y):
        raise ValueError(f"{len(probs)} forecasts for {len(y)} outcomes")
    if (probs < -1e-12).any() or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("every forecast must be non-negative and sum to 1")
    return probs, y


def _one_hot(y: np.ndarray) -> np.ndarray:
    return np.eye(len(OUTCOMES))[y]


def rps(probs, y) -> np.ndarray:
    """Ranked probability score of each forecast: 0 is perfect, lower is better.

    ``1/(K−1) · Σₖ (cumulative forecastₖ − cumulative outcomeₖ)²`` over the first
    K−1 cumulative categories (the last is always 1 − 1).
    """
    probs, y = _check(probs, y)
    cum_forecast = np.cumsum(probs, axis=1)[:, :-1]
    cum_outcome = np.cumsum(_one_hot(y), axis=1)[:, :-1]
    return ((cum_forecast - cum_outcome) ** 2).sum(axis=1) / (len(OUTCOMES) - 1)


def log_loss(probs, y) -> np.ndarray:
    """Negative log probability given to what happened, per forecast."""
    probs, y = _check(probs, y)
    return -np.log(np.clip(probs[np.arange(len(y)), y], _EPS, 1.0))


def brier(probs, y) -> np.ndarray:
    """Multi-class Brier score per forecast: squared error summed over outcomes (0 to 2)."""
    probs, y = _check(probs, y)
    return ((probs - _one_hot(y)) ** 2).sum(axis=1)


METRICS = {"rps": rps, "log_loss": log_loss, "brier": brier}


def scores(probs, y) -> dict[str, float]:
    """Mean of every metric, and how many matches it is over."""
    return {name: float(fn(probs, y).mean()) for name, fn in METRICS.items()} | {"n": len(y)}


def calibration(probs, y, *, bins: int = 10) -> list[dict]:
    """Reliability table, per outcome: predicted probability against observed rate.

    Fixed-width bins over [0, 1]; empty bins are left out. A calibrated forecast has
    ``mean_predicted ≈ observed_rate`` in every bin with enough matches to tell.
    """
    probs, y = _check(probs, y)
    hit = _one_hot(y)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for k, outcome in enumerate(OUTCOMES):
        which = np.clip(np.digitize(probs[:, k], edges[1:-1]), 0, bins - 1)
        for b in range(bins):
            mask = which == b
            if not mask.any():
                continue
            rows.append(
                {
                    "outcome": outcome,
                    # Rounded: linspace gives 0.6000000000000001, which then
                    # appears in the committed report.
                    "bin_low": round(float(edges[b]), 10),
                    "bin_high": round(float(edges[b + 1]), 10),
                    "n": int(mask.sum()),
                    "mean_predicted": float(probs[mask, k].mean()),
                    "observed_rate": float(hit[mask, k].mean()),
                }
            )
    return rows


def draw_summary(probs, y) -> dict[str, float]:
    """How the forecast treats draws (L11), against how often they happened.

    A model can post a respectable log loss while never making a draw the most
    likely outcome; ``draw_most_likely_share`` shows that directly.
    """
    probs, y = _check(probs, y)
    return {
        "actual_draw_rate": float((y == 1).mean()),
        "mean_predicted_draw": float(probs[:, 1].mean()),
        "draw_most_likely_share": float((probs.argmax(axis=1) == 1).mean()),
    }


# --- binary outcomes: shots -------------------------------------------------------------

# Bin edges for xG calibration. Most shots are below 0.1, so fixed 0.1-wide bins would
# put four in five shots into the first one and say nothing about it.
XG_BIN_EDGES = (0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0)


def _check_binary(p, y) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y).astype(int)
    if p.ndim != 1 or p.shape != y.shape:
        raise ValueError(f"need two equal-length 1-d arrays, got {p.shape} and {y.shape}")
    if (p < 0).any() or (p > 1).any():
        raise ValueError("probabilities must lie in [0, 1]")
    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError("outcomes must be 0 or 1")
    return p, y


def binary_log_loss(p, y) -> np.ndarray:
    p, y = _check_binary(p, y)
    p = np.clip(p, _EPS, 1 - _EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def binary_brier(p, y) -> np.ndarray:
    p, y = _check_binary(p, y)
    return (p - y) ** 2


BINARY_METRICS = {"log_loss": binary_log_loss, "brier": binary_brier}


def binary_scores(p, y) -> dict[str, float]:
    """Mean log loss and Brier, AUC, and predicted against observed goal rate."""
    p, y = _check_binary(p, y)
    return {
        "log_loss": float(binary_log_loss(p, y).mean()),
        "brier": float(binary_brier(p, y).mean()),
        "auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
        "mean_predicted": float(p.mean()),
        "observed_rate": float(y.mean()),
        "n": len(y),
    }


def binary_calibration(p, y, *, edges=XG_BIN_EDGES) -> list[dict]:
    """Reliability table: mean predicted against observed rate, per probability bin."""
    p, y = _check_binary(p, y)
    edges = np.asarray(edges, dtype=float)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        mask = which == b
        if mask.any():
            rows.append(
                {
                    "bin_low": round(float(edges[b]), 10),
                    "bin_high": round(float(edges[b + 1]), 10),
                    "n": int(mask.sum()),
                    "mean_predicted": float(p[mask].mean()),
                    "observed_rate": float(y[mask].mean()),
                }
            )
    return rows


@dataclass(frozen=True)
class Difference:
    """Mean per-match score difference, candidate − reference, with a paired interval.

    For every metric here lower is better, so a **negative** difference whose whole
    interval is below zero means the candidate is better.
    """

    metric: str
    candidate: str
    reference: str
    value: float
    low: float
    high: float
    n: int
    resamples: int
    confidence: float
    seed: int
    groups: int | None = None  # resampled units, when they are matches rather than rows

    @property
    def verdict(self) -> str:
        if self.high < 0:
            return f"{self.candidate} is better"
        if self.low > 0:
            return f"{self.reference} is better"
        return "cannot be distinguished"

    def as_dict(self) -> dict:
        return asdict(self) | {"verdict": self.verdict}


def paired_difference(
    candidate_losses,
    reference_losses,
    *,
    metric: str = "rps",
    candidate: str = "candidate",
    reference: str = "reference",
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
    groups=None,
) -> Difference:
    """Paired bootstrap of ``mean(candidate − reference)``.

    Takes per-row losses — the output of :func:`rps`, :func:`binary_log_loss` and
    friends — for the same rows in the same order. Rows are resampled with
    replacement and each resample's mean difference recorded; the interval is the
    percentile interval.

    With ``groups`` (one label per row — a match id for shots) whole groups are
    resampled instead, and each resample's mean is the total difference over the
    rows drawn divided by their number: a cluster bootstrap, for rows that are not
    independent within a group.
    """
    a = np.asarray(candidate_losses, dtype=float)
    b = np.asarray(reference_losses, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError(f"need two equal-length 1-d arrays, got {a.shape} and {b.shape}")
    diff = a - b
    n = len(diff)
    if groups is None:
        sums, counts = diff, np.ones(n)
    else:
        labels, inverse = np.unique(np.asarray(groups), return_inverse=True)
        if len(inverse) != n:
            raise ValueError("groups must have one label per row")
        sums = np.bincount(inverse, weights=diff)
        counts = np.bincount(inverse).astype(float)
    units = len(sums)
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    block = 1_000  # resample in blocks: 10,000 × n at once is needlessly large
    for start in range(0, resamples, block):
        size = min(block, resamples - start)
        draw = rng.integers(0, units, size=(size, units))
        means[start : start + size] = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    tail = 100 * (1 - confidence) / 2
    low, high = np.percentile(means, [tail, 100 - tail])
    return Difference(
        metric=metric,
        candidate=candidate,
        reference=reference,
        value=float(diff.mean()),
        low=float(low),
        high=float(high),
        n=n,
        resamples=resamples,
        confidence=confidence,
        seed=seed,
        groups=None if groups is None else units,
    )
