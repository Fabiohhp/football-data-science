"""Model #3, the shot xG model: select on validation, score the hold-out once.

    uv run python -m football_ml.train shot_xg [--validation-only]

The same method as the match models (see ``train.py``), on the same months:
``splits.aligned`` gives shots the match models' validation and hold-out periods,
and since a match's shots share its date, no match is ever split between training
and test (L5). Candidates are chosen on **log loss** on the validation folds; the
hold-out is scored once per method, and the per-method ledger records it.

Comparisons are paired, and the bootstrap resamples **matches**, not shots: shots
in one match share a game state and a goalkeeper, and treating 1,750 matches'
worth of shots as 43,000 independent rows would make every interval too narrow.

Penalties are not modelled. They sit at one spot with one Understat xG (Part 2
trap 10); the report gives their conversion rate before the hold-out — the constant
a penalty would get — beside Understat's value and the hold-out's actual rate.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, date, datetime

import joblib
import numpy as np
import pandas as pd
import yaml

from football_ml import data, splits
from football_ml import evaluate as ev
from football_ml.models.shot_xg import (
    BASELINE_FEATURES,
    CANDIDATES,
    FEATURES,
    DistanceAngleBaseline,
    ShotCandidate,
    ShotModel,
    Understat,
    UnderstatRecalibrated,
)
from football_ml.train import REPO_ROOT, _fingerprint, _ledger, _to_json, git_state

Forecast = Callable[[np.ndarray, np.ndarray], np.ndarray]

COMPARISONS = [
    ("model", "baseline"),
    ("model", "understat"),
    ("model", "understat_recalibrated"),
    ("understat_recalibrated", "understat"),
]
QUESTION = (
    "How close can a location-and-context xG model get to Understat's, with what Gold "
    "carries? (CLAUDE.md §8 #3)"
)
L10 = (
    "Gold carries no last_action — the action before the shot (cross, through ball, "
    "rebound), which Understat parses and its model uses. Part 1 has it in Silver as "
    "last_action; it is not in Gold's fact_shots. Every gap to Understat below includes "
    "what that column would explain (CLAUDE.md §5, L10)."
)


# The two shots the report maps over the pitch: the commonest kind of attempt, and the
# set-piece header, which a location-only model cannot tell from a shot with the feet.
SURFACE_PROFILES = {
    "Right foot, open play, assisted": {"situation": "OpenPlay", "shot_type": "RightFoot"},
    "Header from a corner, assisted": {"situation": "FromCorner", "shot_type": "Head"},
}


def _pitch() -> dict[str, float]:
    """The pitch geometry feat_shots used, read from the dbt vars — never retyped."""
    vars_ = yaml.safe_load((REPO_ROOT / "dbt" / "dbt_project.yml").read_text())["vars"]
    return {k: float(vars_[k]) for k in ("pitch_length_m", "pitch_width_m", "goal_width_m")}


def surface(model: ShotModel, *, nx: int = 36, ny: int = 51) -> dict:
    """The fitted model's xG over a grid of the attacking third, for two shot profiles.

    Not a refit and not an evaluation: the final model's predictions at chosen points,
    so the report can show what it learned. The geometry is computed exactly as
    feat_shots computes it, from the same dbt vars.
    """
    g = _pitch()
    xs = np.linspace(0.64, 0.995, nx)  # Understat x: towards the goal being attacked
    ys = np.linspace(0.0, 1.0, ny)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    goal_line = (1 - gx) * g["pitch_length_m"]
    lateral = (gy - 0.5) * g["pitch_width_m"]
    half = g["goal_width_m"] / 2
    grid = pd.DataFrame(
        {
            "distance_m": np.sqrt(goal_line**2 + lateral**2).ravel(),
            "angle_rad": np.arctan2(
                g["goal_width_m"] * goal_line, goal_line**2 + lateral**2 - half**2
            ).ravel(),
            "goal_line_m": goal_line.ravel(),
            "lateral_m": lateral.ravel(),
            "minute": 45.0,
            "is_assisted": True,
            "is_home": True,
        }
    )
    profiles = {}
    for label, fixed in SURFACE_PROFILES.items():
        p = model.predict_proba(grid.assign(**fixed)).reshape(nx, ny)
        profiles[label] = [[round(float(v), 4) for v in row] for row in p]
    return {
        "x": [round(float(v), 4) for v in xs],
        "y": [round(float(v), 4) for v in ys],
        "fixed": {"minute": 45, "is_assisted": True, "is_home": True},
        "profiles": profiles,
    }


def walk(folds: list[splits.Fold], forecast: Forecast) -> tuple[np.ndarray, np.ndarray]:
    indices, probs = [], []
    for fold in folds:
        indices.append(fold.test)
        probs.append(forecast(fold.train, fold.test))
    return np.concatenate(indices), np.concatenate(probs)


def _penalties(features: data.Features, holdout_start: date, holdout_end: date) -> dict:
    """Penalty conversion before the hold-out — the constant — and in it."""
    row = features.con.execute(
        """
        select
            count(*) filter (where match_date < ?)                           as before_n,
            avg(is_goal::int) filter (where match_date < ?)                  as before_rate,
            count(*) filter (where match_date >= ? and match_date < ?)       as holdout_n,
            avg(is_goal::int) filter (where match_date >= ? and match_date < ?) as holdout_rate,
            avg(understat_xg)                                                as understat_xg
        from src_shots
        where is_penalty
        """,
        [holdout_start, holdout_start, holdout_start, holdout_end, holdout_start, holdout_end],
    ).fetchone()
    return {
        "before_holdout": {"penalties": row[0], "conversion": row[1]},
        "holdout": {"penalties": row[2], "conversion": row[3]},
        "understat_xg": row[4],
        "note": (
            "Not modelled: every penalty is taken from one spot, and Understat gives each "
            "the same xG. The constant a penalty would get is the conversion rate before "
            "the hold-out."
        ),
    }


def train_shot_xg(*, write: bool = True, validation_only: bool = False) -> dict:
    features = data.open_features()
    matches = (
        features.table("feat_match_outcome")
        .sort_values(["as_of_date", "match_key"], kind="stable")
        .reset_index(drop=True)
    )
    match_plan = splits.plan(matches)
    holdout_end = match_plan.holdout[-1].test_end

    shots = features.table("feat_shots").sort_values(["as_of_date", "shot_key"], kind="stable")
    # The in-progress season is touched by nothing here: it is incomplete.
    shots = shots[pd.to_datetime(shots["as_of_date"]).dt.date < holdout_end].reset_index(drop=True)
    plan = splits.aligned(match_plan, shots["as_of_date"])
    y = shots["target_is_goal"].astype(int).to_numpy()
    groups = shots["understat_match_id"].to_numpy()

    def fitted(make) -> Forecast:
        def forecast(train, test):
            return make().fit(shots.iloc[train], y[train]).predict_proba(shots.iloc[test])

        return forecast

    def candidate(c: ShotCandidate) -> Forecast:
        return fitted(lambda: ShotModel(c))

    methods = {
        "baseline": ([None], lambda _: fitted(DistanceAngleBaseline), lambda _: "baseline"),
        "model": (CANDIDATES, candidate, lambda c: c.name),
        "understat": ([None], lambda _: fitted(Understat), lambda _: "understat"),
        "understat_recalibrated": (
            [None],
            lambda _: fitted(UnderstatRecalibrated),
            lambda _: "understat_recalibrated",
        ),
    }

    # --- 1. validation ----------------------------------------------------------------
    validation, selected = {}, {}
    for method, (grid, build, name) in methods.items():
        rows = []
        for setting in grid:
            idx, p = walk(plan.validation, build(setting))
            rows.append({"name": name(setting)} | ev.binary_scores(p, y[idx]))
        validation[method] = rows
        selected[method] = grid[min(range(len(grid)), key=lambda i: rows[i]["log_loss"])]
    if validation_only:
        features.con.close()
        return {
            "validation": validation,
            "selected": {m: methods[m][2](s) for m, s in selected.items()},
        }

    # --- 2. hold-out ------------------------------------------------------------------
    probs = {}
    for method, (_, build, _) in methods.items():
        idx, probs[method] = walk(plan.holdout, build(selected[method]))
    y_hold, g_hold = y[idx], groups[idx]
    situation = shots["situation"].to_numpy()[idx]
    per_shot = {
        m: {k: fn(p, y_hold) for k, fn in ev.BINARY_METRICS.items()} for m, p in probs.items()
    }
    holdout = {
        "season": plan.holdout_season,
        "shots": len(idx),
        "matches": len(np.unique(g_hold)),
        "scores": {m: ev.binary_scores(p, y_hold) for m, p in probs.items()},
        "comparisons": [
            ev.paired_difference(
                per_shot[a][metric],
                per_shot[b][metric],
                metric=metric,
                candidate=a,
                reference=b,
                groups=g_hold,
            ).as_dict()
            for a, b in COMPARISONS
            for metric in ("log_loss", "brier")
        ],
        "by_situation": {
            s: {
                m: ev.binary_scores(p[situation == s], y_hold[situation == s])
                for m, p in probs.items()
            }
            for s in sorted(set(situation))
        },
        "calibration": {m: ev.binary_calibration(p, y_hold) for m, p in probs.items()},
        "penalties": _penalties(features, plan.holdout_start, holdout_end),
        "missing_from_gold": L10,
    }

    # --- 3. ledger, final fit, card, report --------------------------------------------
    git = git_state()
    recorded = {
        "baseline": {"features": BASELINE_FEATURES},
        "model": asdict(selected["model"]) | {"features": FEATURES},
        "understat": {},
        "understat_recalibrated": {},
    }
    fingerprints = {m: _fingerprint(f"shot_xg.{m}", v, plan) for m, v in recorded.items()}
    report_path = REPO_ROOT / "reports" / "shot_xg.json"
    previous = _load(report_path)
    ledger = _ledger(previous, recorded, fingerprints, git)

    report = {
        "model": "shot_xg",
        "question": QUESTION,
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
            "excluded_live_seasons": plan.live_seasons,
            "selected_on": "validation log loss",
        },
        "features": FEATURES,
        "validation": validation,
        "selected": {m: v | {"name": methods[m][2](selected[m])} for m, v in recorded.items()},
        "holdout": holdout,
        "holdout_evaluations": ledger,
    }

    if write:
        version = f"{date.today():%Y%m%d}-{(git['commit'] or 'nogit')[:7]}"
        final = ShotModel(selected["model"]).fit(shots, y)
        card = {
            "model": "shot_xg",
            "version": version,
            "question": QUESTION,
            "estimator": report["selected"]["model"],
            "inputs": FEATURES,
            "excludes": "penalties and own goals (feat_shots); see the report's penalties block",
            "training_range": {
                "first_match": str(shots["as_of_date"].min().date()),
                "last_match": str(shots["as_of_date"].max().date()),
                "shots": len(shots),
                "matches": int(shots["understat_match_id"].nunique()),
            },
            "snapshot": features.snapshot,
            "citation": features.citation(),
            "selected_on": (
                f"validation folds {plan.validation[0].label} to {plan.validation[-1].label}, "
                "lowest log loss"
            ),
            "holdout_scores": holdout["scores"],
            "holdout_evaluations": ledger["model"],
            "missing_from_gold": L10,
            "git": git,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "limitations": [
                L10,
                "Understat's xG is the benchmark, never an input (L9).",
                "Penalties are not modelled; own goals are not shots.",
            ],
        }
        model_dir = REPO_ROOT / "models" / "shot_xg" / version
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": final}, model_dir / "model.joblib")
        (model_dir / "model_card.json").write_text(_to_json(card))
        report["surface"] = surface(final)
        report_path.parent.mkdir(exist_ok=True)
        report_path.write_text(_to_json(report))

    features.con.close()
    return report


def _load(path):
    return json.loads(path.read_text()) if path.exists() else None


def summary(report: dict) -> str:
    h = report["holdout"]
    lines = [
        f"shot_xg — hold-out {h['season']}, {h['shots']:,} shots in {h['matches']:,} matches",
        *(f"selected {m}: {v['name']}" for m, v in report["selected"].items()),
        "",
        f"{'':24}{'log loss':>10}{'Brier':>9}{'AUC':>8}{'mean p':>9}{'goals':>8}",
    ]
    for m, s in h["scores"].items():
        lines.append(
            f"{m:24}{s['log_loss']:>10.4f}{s['brier']:>9.4f}{s['auc']:>8.4f}"
            f"{s['mean_predicted']:>9.4f}{s['observed_rate']:>8.4f}"
        )
    lines.append("")
    for c in h["comparisons"]:
        lines.append(
            f"{c['candidate']} − {c['reference']} ({c['metric']}): {c['value']:+.5f} "
            f"[{c['low']:+.5f}, {c['high']:+.5f}] → {c['verdict']}"
        )
    pen = h["penalties"]
    lines += [
        "",
        f"penalties: {pen['before_holdout']['conversion']:.3f} before the hold-out "
        f"(n={pen['before_holdout']['penalties']}), {pen['holdout']['conversion']:.3f} in it "
        f"(n={pen['holdout']['penalties']}); Understat {pen['understat_xg']:.3f}",
        "",
        "hold-out evaluations recorded, per method: "
        + str({m: len(v) for m, v in report["holdout_evaluations"].items()}),
    ]
    return "\n".join(lines)
