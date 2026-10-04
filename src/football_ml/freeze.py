"""Freeze the match models, so a later snapshot can score them honestly (CLAUDE.md §7).

    uv run python -m football_ml.freeze

Prospective evaluation is the one test in this repository that no care in the split
could have fooled: a model fixed at one moment, scored on matches played after it.
This module makes that moment a record rather than a claim.

**What is frozen.** All five match methods — base rates, Elo, model #1 and both
Dixon-Coles variants — with the settings chosen on validation and recorded in
``reports/match_outcome.json``. Nothing is re-selected here. Each is refitted on
every match in the current snapshot, as a model put into use would be, and their
fitted parameters are then fixed: Dixon-Coles keeps the club strengths it had on
the day, model #1 its coefficients, Elo its slope and cut-points. What still moves
is what is a *feature*: model #1 reads form rebuilt from the later snapshot, and Elo
its ratings, which by construction take only results from earlier dates (L4).

**The record.** ``models/frozen/<id>/freeze_card.json`` — committed — holds the
moment, the commit, the snapshot, the settings, the training range, and the
SHA-256 of the fitted bundle. The bundle itself is gitignored with every other
binary; the scorer refuses one whose hash differs from the card's, so the thing
scored is provably the thing frozen.

**Which matches it may be scored on.** Only those dated after **both** the last
match in the snapshot it was fitted on **and** the day it was frozen
(``scores_matches_after``). The first is what the model cannot have seen; the second
is what nobody could have seen either — a match already played when the model was
frozen is not a prospective test, even if it was not in the data yet.

**It refuses a dirty working tree.** A card's commit hash is only evidence if that
commit contains the code that fitted the model. Commit, then freeze, then commit the
card.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib

from football_ml import data
from football_ml import evaluate as ev
from football_ml.baselines import BaseRates, EloForecast, EloSettings, pre_match_elo
from football_ml.models.dixon_coles import DixonColes, DixonColesSettings
from football_ml.models.match_outcome import Candidate, FormModel, feature_columns
from football_ml.train import REPO_ROOT, _to_json, git_state

FROZEN_DIR = REPO_ROOT / "models" / "frozen"
MATCH_REPORT = REPO_ROOT / "reports" / "match_outcome.json"
BUNDLE = "frozen.joblib"
CARD = "freeze_card.json"


class FreezeRefused(RuntimeError):
    """The freeze would not be a trustworthy record; the message says why."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _settings(selected: dict) -> dict:
    """The validation-chosen settings of each method, rebuilt from the report."""

    def strip(d, *drop):
        return {k: v for k, v in d.items() if k not in {"name", *drop}}

    return {
        "elo": EloSettings(**strip(selected["elo"])),
        "model": Candidate(**strip(selected["model"], "features")),
        "dc_goals": DixonColesSettings(**strip(selected["dc_goals"])),
        "dc_xg": DixonColesSettings(**strip(selected["dc_xg"])),
    }


def freeze(
    *,
    frozen_at: datetime | None = None,
    allow_dirty: bool = False,
    report_path: Path = MATCH_REPORT,
    frozen_dir: Path = FROZEN_DIR,
) -> Path:
    """Fit and freeze every match method; return the path of the freeze card.

    ``frozen_at`` and ``allow_dirty`` exist for the tests, which freeze a synthetic
    snapshot at a chosen moment from an uncommitted tree. A real freeze passes neither.
    """
    git = git_state()
    if git["commit"] is None:
        raise FreezeRefused("Not a git repository: a freeze must name the commit it came from.")
    if git["dirty"] and not allow_dirty:
        raise FreezeRefused(
            "The working tree has uncommitted changes, so the card's commit hash would not "
            "contain the code that fitted the model. Commit first, then freeze."
        )
    if not report_path.exists():
        raise FreezeRefused(
            f"No {report_path.name}: settings come from validation, so train first:\n"
            "  uv run python -m football_ml.train match_outcome"
        )
    report = json.loads(report_path.read_text())
    settings = _settings(report["selected"])

    features = data.open_features()
    frame = (
        features.table("feat_match_outcome")
        .sort_values(["as_of_date", "match_key"], kind="stable")
        .reset_index(drop=True)
    )
    predictors = feature_columns(frame)
    if predictors != report["selected"]["model"]["features"]:
        raise FreezeRefused(
            "feat_match_outcome's features differ from the ones model #1's settings were "
            "chosen with. Re-run training (and its validation) before freezing."
        )
    y = ev.outcome_index(frame["result"])
    elo_settings = settings["elo"]

    bundle = {
        "base_rates": BaseRates().fit(frame, y),
        "elo_settings": elo_settings,
        "elo": EloForecast(elo_settings).fit(pre_match_elo(frame, elo_settings), y),
        "model": FormModel(settings["model"]).fit(frame, y),
        "dc_goals": DixonColes(settings["dc_goals"]).fit(frame),
        "dc_xg": DixonColes(settings["dc_xg"]).fit(frame),
    }

    frozen_at = (frozen_at or datetime.now(UTC)).astimezone(UTC)
    last_match = frame["as_of_date"].max().date()
    scores_after = max(last_match, frozen_at.date())
    freeze_id = f"{frozen_at:%Y%m%dT%H%MZ}-{git['commit'][:7]}"
    target = frozen_dir / freeze_id
    target.mkdir(parents=True, exist_ok=False)
    joblib.dump(bundle, target / BUNDLE)

    card = {
        "freeze_id": freeze_id,
        "frozen_at": frozen_at.isoformat(timespec="seconds"),
        "scores_matches_after": str(scores_after),
        "rule": (
            "Scored only on matches dated after both the last match it was fitted on and "
            "the day it was frozen."
        ),
        "git": git,
        "snapshot": features.snapshot,
        "citation": features.citation(),
        "training_range": {
            "first_match": str(frame["as_of_date"].min().date()),
            "last_match": str(last_match),
            "matches": len(frame),
        },
        "methods": {
            "base_rates": "league H/D/A frequencies of the training matches",
            "elo": report["selected"]["elo"],
            "model": report["selected"]["model"],
            "dc_goals": report["selected"]["dc_goals"],
            "dc_xg": report["selected"]["dc_xg"],
        },
        "settings_from": {
            "report": (
                str(report_path.relative_to(REPO_ROOT))
                if report_path.is_relative_to(REPO_ROOT)
                else str(report_path)
            ),
            "holdout_evaluations": report.get("holdout_evaluations"),
        },
        "artifact": {"file": BUNDLE, "sha256": sha256(target / BUNDLE)},
    }
    (target / CARD).write_text(_to_json(card))
    features.con.close()
    return target / CARD


def latest(frozen_dir: Path = FROZEN_DIR) -> Path | None:
    cards = sorted(frozen_dir.glob(f"*/{CARD}"))
    return cards[-1] if cards else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the match models for prospective scoring.")
    parser.parse_args()
    try:
        card = freeze()
    except FreezeRefused as exc:
        raise SystemExit(f"freeze refused: {exc}") from None
    payload = json.loads(card.read_text())
    print(f"frozen {payload['freeze_id']}")
    print(
        f"  fitted on {payload['training_range']['matches']:,} matches up to "
        f"{payload['training_range']['last_match']}"
    )
    print(f"  will be scored on matches after {payload['scores_matches_after']}")
    print(f"  card: {card.relative_to(REPO_ROOT)} — commit it")


if __name__ == "__main__":
    main()
