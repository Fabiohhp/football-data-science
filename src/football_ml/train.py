"""Fit → select on validation → score the hold-out once → model card and report.

    uv run python -m football_ml.train match_outcome

The order of operations is the method (CLAUDE.md §7):

1. **Validation.** Every candidate — each Elo ``k``, each model setting — is walked
   forward month by month through the validation period, and the one with the
   lowest ranked probability score is chosen *per method*. Every validation fold
   trains and tests on matches before the hold-out's first date (tested in
   tests/test_splits.py); Elo ratings are computed over the whole frame, but a
   rating depends only on earlier dates, so a hold-out result cannot reach one.
   ``--validation-only`` stops here, which is how the pipeline is developed without
   spending the hold-out.
2. **Hold-out.** With those settings frozen, base rates, Elo and the model are
   walked forward through the hold-out season, refitted each month on everything
   before it. Scored once, compared pairwise with bootstrap intervals.
3. **Final fit.** The chosen model is refitted on everything up to the end of the
   hold-out and saved with its model card — the frozen model phase 5 scores
   prospectively.

The in-progress season is dropped before step 1 and touched by none of it.

**The hold-out is scored once, and the report says how many times it was.**
``reports/match_outcome.json`` keeps a ledger of hold-out evaluations, keyed by a
fingerprint of everything that decides the hold-out score (settings, features,
folds). Re-running unchanged reproduces the same entry; running with anything
changed — a new feature, a new grid — appends a second, and the report then says
the hold-out has been looked at twice. That does not stop anyone; it makes it
visible, which is what L6 asks of a report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from football_ml import data, splits
from football_ml import evaluate as ev
from football_ml.baselines import BaseRates, EloForecast, EloSettings, pre_match_elo
from football_ml.models.match_outcome import CANDIDATES, Candidate, FormModel, feature_columns

REPO_ROOT = Path(__file__).resolve().parents[2]
ELO_GRID = [EloSettings(k=k) for k in (10.0, 15.0, 20.0, 30.0, 40.0)]
COMPARISONS = [("model", "elo"), ("model", "base_rates"), ("elo", "base_rates")]

Forecast = Callable[[np.ndarray, np.ndarray], np.ndarray]


def git_state() -> dict:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def walk(folds: list[splits.Fold], forecast: Forecast) -> tuple[np.ndarray, np.ndarray]:
    """Run ``forecast(train, test)`` on every fold; return test indices and probabilities."""
    indices, probs = [], []
    for fold in folds:
        indices.append(fold.test)
        probs.append(forecast(fold.train, fold.test))
    return np.concatenate(indices), np.vstack(probs)


def _rounded(obj, places: int = 5):
    """Round every float, so a re-run with the same inputs writes the same file."""
    if isinstance(obj, float):
        return round(obj, places)
    if isinstance(obj, dict):
        return {k: _rounded(v, places) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_rounded(v, places) for v in obj]
    return obj


def _to_json(obj) -> str:
    # ensure_ascii=False keeps "—" and "§" readable on GitHub instead of \u2014.
    return json.dumps(_rounded(obj), indent=2, ensure_ascii=False) + "\n"


def train_match_outcome(*, write: bool = True, validation_only: bool = False) -> dict:
    features = data.open_features()
    frame = (
        features.table("feat_match_outcome")
        .sort_values(["as_of_date", "match_key"], kind="stable")
        .reset_index(drop=True)
    )
    plan = splits.plan(frame)

    # Drop the live season before anything else sees it (phase 5's prospective test).
    holdout_end = plan.holdout[-1].test_end
    frame = frame[pd.to_datetime(frame["as_of_date"]).dt.date < holdout_end].reset_index(drop=True)
    plan = splits.plan(frame)
    y = ev.outcome_index(frame["result"])
    predictors = feature_columns(frame)

    elo_by_settings = {s: pre_match_elo(frame, s) for s in ELO_GRID}

    def base_rates(train, test):
        return BaseRates().fit(frame.iloc[train], y[train]).predict_proba(frame.iloc[test])

    def elo(settings: EloSettings) -> Forecast:
        ratings = elo_by_settings[settings]

        def forecast(train, test):
            model = EloForecast(settings).fit(ratings.iloc[train], y[train])
            return model.predict_proba(ratings.iloc[test])

        return forecast

    def form(candidate: Candidate) -> Forecast:
        def forecast(train, test):
            model = FormModel(candidate).fit(frame.iloc[train], y[train])
            return model.predict_proba(frame.iloc[test])

        return forecast

    # --- 1. validation: choose settings per method -----------------------------------
    def validate(name: str, forecast: Forecast) -> dict:
        idx, probs = walk(plan.validation, forecast)
        return {"name": name, **ev.scores(probs, y[idx])}

    validation = {
        "base_rates": [validate("base_rates", base_rates)],
        "elo": [validate(f"elo(k={s.k:g})", elo(s)) | {"k": s.k} for s in ELO_GRID],
        "model": [validate(c.name, form(c)) | {"candidate": asdict(c)} for c in CANDIDATES],
    }
    best_elo = EloSettings(k=min(validation["elo"], key=lambda r: r["rps"])["k"])
    best_row = min(validation["model"], key=lambda r: r["rps"])
    best = Candidate(**best_row["candidate"])
    if validation_only:
        features.con.close()
        return {"validation": validation, "selected": {"elo": best_elo, "model": best.name}}

    # --- 2. hold-out: frozen settings, scored once ------------------------------------
    methods: dict[str, Forecast] = {
        "base_rates": base_rates,
        "elo": elo(best_elo),
        "model": form(best),
    }
    holdout_probs = {}
    for name, forecast in methods.items():
        idx, holdout_probs[name] = walk(plan.holdout, forecast)
    y_hold = y[idx]
    short = frame["short_history"].to_numpy()[idx].astype(bool)

    per_match = {
        name: {m: fn(p, y_hold) for m, fn in ev.METRICS.items()}
        for name, p in holdout_probs.items()
    }
    holdout = {
        "season": plan.holdout_season,
        "n": len(idx),
        "scores": {name: ev.scores(p, y_hold) for name, p in holdout_probs.items()},
        "comparisons": [
            ev.paired_difference(
                per_match[a][metric], per_match[b][metric], metric=metric, candidate=a, reference=b
            ).as_dict()
            for a, b in COMPARISONS
            for metric in ("rps", "log_loss")
        ],
        "by_history": {
            label: {name: ev.scores(p[mask], y_hold[mask]) for name, p in holdout_probs.items()}
            for label, mask in [("short_history", short), ("full_history", ~short)]
        },
        "draws": {name: ev.draw_summary(p, y_hold) for name, p in holdout_probs.items()},
        "calibration": {name: ev.calibration(p, y_hold) for name, p in holdout_probs.items()},
    }

    # --- 3. final fit, card, report ---------------------------------------------------
    git = git_state()
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "elo": asdict(best_elo),
                "model": asdict(best),
                "features": predictors,
                "holdout_folds": [f.label for f in plan.holdout],
                "validation_folds": [f.label for f in plan.validation],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]

    report_path = REPO_ROOT / "reports" / "match_outcome.json"
    ledger = []
    if report_path.exists():
        ledger = json.loads(report_path.read_text()).get("holdout_evaluations", [])
    if not any(entry["fingerprint"] == fingerprint for entry in ledger):
        ledger.append(
            {"fingerprint": fingerprint, "git_commit": git["commit"], "date": str(date.today())}
        )

    version = f"{date.today():%Y%m%d}-{(git['commit'] or 'nogit')[:7]}"
    training_rows = frame  # everything up to the end of the hold-out
    card = {
        "model": "match_outcome",
        "version": version,
        "question": (
            "Can pre-match form — goals, xG, xPTS — predict H/D/A better than league "
            "base rates and Elo? (CLAUDE.md §8 #1)"
        ),
        "estimator": asdict(best) | {"name": best.name},
        "features": predictors + ["competition_code"],
        "training_range": {
            "first_match": str(training_rows["as_of_date"].min().date()),
            "last_match": str(training_rows["as_of_date"].max().date()),
            "matches": len(training_rows),
        },
        "snapshot": features.snapshot,
        "citation": features.citation(),
        "selected_on": (
            f"validation folds {plan.validation[0].label} to {plan.validation[-1].label}, "
            "lowest ranked probability score"
        ),
        "holdout_scores": holdout["scores"],
        "baselines": {"elo": asdict(best_elo)},
        "git": git,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "limitations": [
            "Compared with league base rates and Elo only. Gold holds no betting odds, "
            "so there is no market benchmark and no claim to beat one.",
            "What the model uses is not what wins matches: no causal reading of its "
            "coefficients or importances.",
            "History starts in August 2024; a club's form counts from there (L7).",
        ],
    }
    report = {
        "model": "match_outcome",
        "citation": features.citation(),
        "git": git,
        "plan": {
            "validation_folds": [
                {"month": f.label, "train": len(f.train), "test": len(f.test)}
                for f in plan.validation
            ],
            "holdout_folds": [
                {"month": f.label, "train": len(f.train), "test": len(f.test)} for f in plan.holdout
            ],
            "holdout_season": plan.holdout_season,
            "excluded_live_seasons": splits.plan(features.table("feat_match_outcome")).live_seasons,
        },
        "features": predictors,
        "validation": validation,
        "selected": {"elo": asdict(best_elo), "model": asdict(best) | {"name": best.name}},
        "holdout": holdout,
        "holdout_evaluations": ledger,
    }

    if write:
        final = FormModel(best).fit(frame, y)
        model_dir = REPO_ROOT / "models" / "match_outcome" / version
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": final,
                "elo_settings": best_elo,
                "elo_forecast": EloForecast(best_elo).fit(elo_by_settings[best_elo], y),
            },
            model_dir / "model.joblib",
        )
        (model_dir / "model_card.json").write_text(_to_json(card))
        report_path.parent.mkdir(exist_ok=True)
        report_path.write_text(_to_json(report))

    features.con.close()
    return report


def summary(report: dict) -> str:
    h = report["holdout"]
    lines = [
        f"match_outcome — hold-out {h['season']}, {h['n']:,} matches",
        f"selected: {report['selected']['model']['name']}, "
        f"elo k={report['selected']['elo']['k']:g}",
        "",
        f"{'':12}{'RPS':>9}{'log loss':>10}{'Brier':>9}",
    ]
    for name, s in h["scores"].items():
        lines.append(f"{name:12}{s['rps']:>9.4f}{s['log_loss']:>10.4f}{s['brier']:>9.4f}")
    lines.append("")
    for c in h["comparisons"]:
        lines.append(
            f"{c['candidate']} − {c['reference']} ({c['metric']}): {c['value']:+.4f} "
            f"[{c['low']:+.4f}, {c['high']:+.4f}] → {c['verdict']}"
        )
    lines.append("")
    for name, d in h["draws"].items():
        lines.append(
            f"draws, {name}: actual {d['actual_draw_rate']:.3f}, mean predicted "
            f"{d['mean_predicted_draw']:.3f}, most likely in {d['draw_most_likely_share']:.1%}"
        )
    looks = len(report["holdout_evaluations"])
    lines.append(f"\nhold-out evaluations recorded: {looks}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model", choices=["match_outcome"])
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="score the candidates on the validation folds and stop: the hold-out is not touched",
    )
    args = parser.parse_args()
    if args.validation_only:
        result = train_match_outcome(write=False, validation_only=True)
        for rows in result["validation"].values():
            for r in rows:
                print(
                    f"{r['name']:60} RPS {r['rps']:.4f}  log loss {r['log_loss']:.4f}  n {r['n']}"
                )
        print("selected:", result["selected"])
        return
    print(summary(train_match_outcome()))


if __name__ == "__main__":
    main()
