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
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

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
) -> Difference:
    """Paired bootstrap over matches of ``mean(candidate − reference)``.

    Takes per-match losses — the output of :func:`rps` and friends — for the same
    matches in the same order. Matches are resampled with replacement and each
    resample's mean difference recorded; the interval is the percentile interval.
    """
    a = np.asarray(candidate_losses, dtype=float)
    b = np.asarray(reference_losses, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError(f"need two equal-length 1-d arrays, got {a.shape} and {b.shape}")
    diff = a - b
    n = len(diff)
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    block = 1_000  # resample in blocks: 10,000 × n at once is needlessly large
    for start in range(0, resamples, block):
        size = min(block, resamples - start)
        means[start : start + size] = diff[rng.integers(0, n, size=(size, n))].mean(axis=1)
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
    )
