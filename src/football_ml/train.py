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
from football_ml.models.dixon_coles import DixonColes, DixonColesSettings
from football_ml.models.match_outcome import CANDIDATES, Candidate, FormModel, feature_columns

REPO_ROOT = Path(__file__).resolve().parents[2]
ELO_GRID = [EloSettings(k=k) for k in (10.0, 15.0, 20.0, 30.0, 40.0)]
# Model #2's grid: time decay per day (none; half-lives of ~350, ~140 and ~70 days) and
# the ridge penalty on attacks and defences. The same grid for both targets. The first
# grid (decay up to 0.005, penalty 1 or 4) had both variants choosing an edge on
# validation, so it was widened once each way before the hold-out was touched.
DC_GRIDS = {
    target: [
        DixonColesSettings(target=target, xi=xi, penalty=penalty)
        for xi in (0.0, 0.002, 0.005, 0.01)
        for penalty in (0.25, 1.0, 4.0, 16.0)
    ]
    for target in ("goals", "xg")
}
COMPARISONS = [
    ("model", "elo"),
    ("model", "base_rates"),
    ("elo", "base_rates"),
    ("dc_goals", "elo"),
    ("dc_goals", "model"),
    ("dc_xg", "dc_goals"),
    ("dc_xg", "elo"),
]

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


def _fingerprint(method: str, settings: dict, plan: splits.Plan) -> str:
    """Everything that decides one method's hold-out score, hashed."""
    payload = {
        "method": method,
        "settings": settings,
        "holdout_folds": [f.label for f in plan.holdout],
        "validation_folds": [f.label for f in plan.validation],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _ledger(
    previous: dict | None, recorded: dict[str, dict], fingerprints: dict[str, str], git: dict
) -> dict:
    """Per-method record of hold-out evaluations, carried from the last report.

    A method whose fingerprint is already recorded is being reproduced, not looked at
    again. A new fingerprint for a method already on record is a second look, and the
    report shows it. The first report kept one combined list for base rates, Elo and
    model #1; it is carried over onto those methods — as the same evaluation if their
    settings are unchanged, and as a separate, earlier one if they are not, so that a
    look at the hold-out is never dropped from the record.
    """
    previous = previous or {}
    old = previous.get("holdout_evaluations", {})
    ledger: dict[str, list[dict]] = {}
    if isinstance(old, list):  # the combined format of the first report
        old_selected = previous.get("selected", {})
        for method in ("base_rates", "elo", "model"):
            then = {k: v for k, v in old_selected.get(method, {}).items() if k != "name"}
            if method == "model":  # the first report kept the feature list apart
                then["features"] = previous.get("features")
            unchanged = then == recorded[method]
            entries = [
                entry
                | {
                    "fingerprint": fingerprints[method] if unchanged else entry["fingerprint"],
                    "note": "carried over from the combined ledger of the first report",
                }
                for entry in old
            ]
            ledger[method] = entries
    else:
        ledger = {m: list(v) for m, v in old.items()}

    for method, fingerprint in fingerprints.items():
        entries = ledger.setdefault(method, [])
        if not any(e["fingerprint"] == fingerprint for e in entries):
            entries.append(
                {"fingerprint": fingerprint, "git_commit": git["commit"], "date": str(date.today())}
            )
    return ledger


QUESTIONS = {
    "model": "Can pre-match form — goals, xG, xPTS — predict H/D/A better than league "
    "base rates and Elo? (CLAUDE.md §8 #1)",
    "dc_goals": "Does modelling goals per side (Dixon-Coles) beat modelling the result "
    "directly? (CLAUDE.md §8 #2)",
    "dc_xg": "Does fitting the Dixon-Coles scoring rates to xG instead of goals help? "
    "(CLAUDE.md §8 #2)",
}
CARD_NAMES = {"model": "match_outcome", "dc_goals": "dixon_coles_goals", "dc_xg": "dixon_coles_xg"}
LIMITATIONS = [
    "Compared with league base rates and Elo only. Gold holds no betting odds, so there "
    "is no market benchmark and no claim to beat one.",
    "What a model uses is not what wins matches: no causal reading of its parameters, "
    "coefficients or importances.",
    "History starts in August 2024; a club's record counts from there (L7).",
]


def train_match_outcome(*, write: bool = True, validation_only: bool = False) -> dict:
    features = data.open_features()
    frame = (
        features.table("feat_match_outcome")
        .sort_values(["as_of_date", "match_key"], kind="stable")
        .reset_index(drop=True)
    )
    live_seasons = splits.plan(frame).live_seasons
    plan = splits.plan(frame)

    # Drop the live season before anything else sees it (phase 5's prospective test).
    holdout_end = plan.holdout[-1].test_end
    frame = frame[pd.to_datetime(frame["as_of_date"]).dt.date < holdout_end].reset_index(drop=True)
    plan = splits.plan(frame)
    y = ev.outcome_index(frame["result"])
    predictors = feature_columns(frame)

    elo_by_settings = {s: pre_match_elo(frame, s) for s in ELO_GRID}

    def base_rates(_settings=None) -> Forecast:
        def forecast(train, test):
            return BaseRates().fit(frame.iloc[train], y[train]).predict_proba(frame.iloc[test])

        return forecast

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

    def dixon_coles(settings: DixonColesSettings) -> Forecast:
        def forecast(train, test):
            return DixonColes(settings).fit(frame.iloc[train]).predict_proba(frame.iloc[test])

        return forecast

    # name -> (grid, how to build a forecast, how to name a setting, how to record it)
    methods = {
        "base_rates": ([None], base_rates, lambda s: "base_rates", lambda s: {}),
        "elo": (ELO_GRID, elo, lambda s: f"elo(k={s.k:g})", asdict),
        "model": (CANDIDATES, form, lambda s: s.name, asdict),
        "dc_goals": (DC_GRIDS["goals"], dixon_coles, lambda s: s.name, asdict),
        "dc_xg": (DC_GRIDS["xg"], dixon_coles, lambda s: s.name, asdict),
    }

    # --- 1. validation: choose settings per method -----------------------------------
    validation, selected = {}, {}
    for method, (grid, build, name, record) in methods.items():
        rows = []
        for setting in grid:
            idx, probs = walk(plan.validation, build(setting))
            rows.append(
                {"name": name(setting), "settings": record(setting)} | ev.scores(probs, y[idx])
            )
        validation[method] = rows
        best_i = min(range(len(grid)), key=lambda i: rows[i]["rps"])
        selected[method] = grid[best_i]
    if validation_only:
        features.con.close()
        return {
            "validation": validation,
            "selected": {m: methods[m][2](s) for m, s in selected.items()},
        }

    # --- 2. hold-out: frozen settings, scored once ------------------------------------
    holdout_probs = {}
    for method, (_, build, _, _) in methods.items():
        idx, holdout_probs[method] = walk(plan.holdout, build(selected[method]))
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

    # --- 3. ledger, final fits, cards, report -----------------------------------------
    git = git_state()
    recorded = {m: methods[m][3](s) for m, s in selected.items()}
    recorded["model"] = recorded["model"] | {"features": predictors}
    fingerprints = {m: _fingerprint(m, v, plan) for m, v in recorded.items()}

    report_path = REPO_ROOT / "reports" / "match_outcome.json"
    previous = json.loads(report_path.read_text()) if report_path.exists() else None
    ledger = _ledger(previous, recorded, fingerprints, git)

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
            "excluded_live_seasons": live_seasons,
        },
        "features": predictors,
        "validation": validation,
        "selected": {m: v | {"name": methods[m][2](selected[m])} for m, v in recorded.items()},
        "holdout": holdout,
        "holdout_evaluations": ledger,
    }

    if write:
        version = f"{date.today():%Y%m%d}-{(git['commit'] or 'nogit')[:7]}"
        finals = {
            "model": {
                "model": FormModel(selected["model"]).fit(frame, y),
                "elo_settings": selected["elo"],
                "elo_forecast": EloForecast(selected["elo"]).fit(
                    elo_by_settings[selected["elo"]], y
                ),
            },
            "dc_goals": {"model": DixonColes(selected["dc_goals"]).fit(frame)},
            "dc_xg": {"model": DixonColes(selected["dc_xg"]).fit(frame)},
        }
        for method, bundle in finals.items():
            card = {
                "model": CARD_NAMES[method],
                "version": version,
                "question": QUESTIONS[method],
                "estimator": report["selected"][method],
                "inputs": (
                    predictors + ["competition_code"]
                    if method == "model"
                    else ["home_team_key", "away_team_key", "competition_code", "season"]
                    + (["target_home_xg", "target_away_xg"] if method == "dc_xg" else [])
                    + ["target_home_goals", "target_away_goals"]
                ),
                "training_range": {
                    "first_match": str(frame["as_of_date"].min().date()),
                    "last_match": str(frame["as_of_date"].max().date()),
                    "matches": len(frame),
                },
                "snapshot": features.snapshot,
                "citation": features.citation(),
                "selected_on": (
                    f"validation folds {plan.validation[0].label} to "
                    f"{plan.validation[-1].label}, lowest ranked probability score"
                ),
                "holdout_scores": {k: holdout["scores"][k] for k in ("base_rates", "elo", method)},
                "holdout_evaluations": ledger[method],
                "git": git,
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "limitations": LIMITATIONS,
            }
            model_dir = REPO_ROOT / "models" / CARD_NAMES[method] / version
            model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(bundle, model_dir / "model.joblib")
            (model_dir / "model_card.json").write_text(_to_json(card))
        report_path.parent.mkdir(exist_ok=True)
        report_path.write_text(_to_json(report))

    features.con.close()
    return report


def summary(report: dict) -> str:
    h = report["holdout"]
    lines = [
        f"match_outcome — hold-out {h['season']}, {h['n']:,} matches",
        *(f"selected {m}: {v['name']}" for m, v in report["selected"].items()),
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
    looks = {m: len(v) for m, v in report["holdout_evaluations"].items()}
    lines.append(f"\nhold-out evaluations recorded, per method: {looks}")
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
        for method, name in result["selected"].items():
            print(f"selected {method}: {name}")
        return
    print(summary(train_match_outcome()))


if __name__ == "__main__":
    main()
