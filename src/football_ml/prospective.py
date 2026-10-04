"""Score a frozen model on matches played after it was frozen (CLAUDE.md §7).

    uv run python -m football_ml.prospective [--freeze <id>]

Run it after a newer snapshot has reached Part 2 and the features have been rebuilt
from it. It loads the latest freeze (or the one named), checks the fitted bundle is
byte-for-byte the one the card describes, and scores every match dated after the
card's ``scores_matches_after`` — matches no model here, and nobody choosing its
settings, could have seen.

Nothing is refitted. Features for the new matches come from the rebuilt
feat_match_outcome, which is point-in-time by the same tests as always (L4); Elo's
ratings are recomputed through the new results, each rating from strictly earlier
dates only. The frozen parameters do not move.

It writes ``reports/prospective_<freeze id>.json`` — scores, paired comparisons with
match-level intervals, draws and calibration — and the per-match predictions for the
export. With no newer snapshot, or no match yet after the freeze, it says so and
writes nothing.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from football_ml import data, predictions
from football_ml import evaluate as ev
from football_ml.baselines import pre_match_elo
from football_ml.freeze import FROZEN_DIR, latest, sha256
from football_ml.train import REPO_ROOT, _to_json, git_state

METHODS = ["base_rates", "elo", "model", "dc_goals", "dc_xg"]
COMPARISONS = [
    ("dc_xg", "elo"),
    ("dc_goals", "elo"),
    ("model", "elo"),
    ("elo", "base_rates"),
]


class ProspectiveRefused(RuntimeError):
    """The frozen artefact cannot be trusted, or there is nothing it may be scored on."""


def _load(card_path: Path) -> tuple[dict, dict]:
    card = json.loads(card_path.read_text())
    bundle_path = card_path.parent / card["artifact"]["file"]
    if not bundle_path.exists():
        raise ProspectiveRefused(
            f"{bundle_path} is missing. The bundle is gitignored and lives only on the "
            "machine that froze it; the card alone cannot be scored."
        )
    actual = sha256(bundle_path)
    if actual != card["artifact"]["sha256"]:
        raise ProspectiveRefused(
            f"{bundle_path.name} does not match its freeze card (sha256 {actual[:12]}… "
            f"against {card['artifact']['sha256'][:12]}…). Whatever it is, it is not the "
            "model that was frozen."
        )
    return card, joblib.load(bundle_path)


def score(
    card_path: Path | None = None, *, write: bool = True, frozen_dir: Path = FROZEN_DIR
) -> dict:
    card_path = card_path or latest(frozen_dir)
    if card_path is None:
        raise ProspectiveRefused("Nothing is frozen yet:\n  uv run python -m football_ml.freeze")
    card, bundle = _load(card_path)

    features = data.open_features()
    status = {"freeze_id": card["freeze_id"], "frozen_on": card["citation"]}
    try:
        exported = datetime.fromisoformat(features.snapshot["exported_at"])
        if exported <= datetime.fromisoformat(card["snapshot"]["exported_at"]):
            return status | {
                "status": "waiting",
                "reason": "the features are built from the snapshot the model was frozen on, "
                "or an older one — export a newer snapshot in Part 2 and rebuild",
            }

        frame = (
            features.table("feat_match_outcome")
            .sort_values(["as_of_date", "match_key"], kind="stable")
            .reset_index(drop=True)
        )
        after = pd.Timestamp(card["scores_matches_after"])
        target = frame["as_of_date"] > after
        if not target.any():
            return status | {
                "status": "waiting",
                "reason": f"no match yet after {card['scores_matches_after']} in "
                f"{features.citation()}",
            }
        trained_until = pd.Timestamp(card["training_range"]["last_match"])
        assert (frame.loc[target, "as_of_date"] > trained_until).all()  # never a seen match

        rows = frame[target].reset_index(drop=True)
        ratings = pre_match_elo(frame, bundle["elo_settings"])[target.to_numpy()]
        probs = {
            "base_rates": bundle["base_rates"].predict_proba(rows),
            "elo": bundle["elo"].predict_proba(ratings.reset_index(drop=True)),
            "model": bundle["model"].predict_proba(rows),
            "dc_goals": bundle["dc_goals"].predict_proba(rows),
            "dc_xg": bundle["dc_xg"].predict_proba(rows),
        }
        y = ev.outcome_index(rows["result"])
        per_match = {m: {k: fn(p, y) for k, fn in ev.METRICS.items()} for m, p in probs.items()}

        report = status | {
            "status": "scored",
            "scored_on": features.citation(),
            "snapshot": features.snapshot,
            "scores_matches_after": card["scores_matches_after"],
            "matches": len(rows),
            "date_range": {
                "first": str(rows["as_of_date"].min().date()),
                "last": str(rows["as_of_date"].max().date()),
            },
            "by_league": rows["competition_code"].value_counts().sort_index().to_dict(),
            "scores": {m: ev.scores(p, y) for m, p in probs.items()},
            "comparisons": [
                ev.paired_difference(
                    per_match[a][metric], per_match[b][metric], metric=metric,
                    candidate=a, reference=b,
                ).as_dict()
                for a, b in COMPARISONS
                for metric in ("rps", "log_loss")
            ],  # fmt: skip
            "draws": {m: ev.draw_summary(p, y) for m, p in probs.items()},
            "calibration": {m: ev.calibration(p, y) for m, p in probs.items()},
            "scored_with": git_state(),
            "caveat": (
                f"{len(rows)} matches. Prospective intervals are wide until a season's "
                "worth has accumulated; a verdict of 'cannot be distinguished' this early is "
                "the expected reading, not a finding."
            ),
        }
        if write:
            (REPO_ROOT / "reports" / f"prospective_{card['freeze_id']}.json").write_text(
                _to_json(report)
            )
            predictions.write(f"prospective_{card['freeze_id']}", predictions.table(rows, probs))
        return report
    finally:
        features.con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a frozen model on later matches.")
    parser.add_argument("--freeze", help="freeze id (default: the latest)")
    args = parser.parse_args()
    card = FROZEN_DIR / args.freeze / "freeze_card.json" if args.freeze else None
    try:
        report = score(card)
    except ProspectiveRefused as exc:
        raise SystemExit(f"prospective scoring refused: {exc}") from None
    print(f"freeze {report['freeze_id']}: {report['status']}")
    if report["status"] != "scored":
        print(f"  {report['reason']}")
        return
    print(
        f"  {report['matches']} matches, {report['date_range']['first']} to "
        f"{report['date_range']['last']}"
    )
    for m, s in report["scores"].items():
        print(f"  {m:12} RPS {s['rps']:.4f}  log loss {s['log_loss']:.4f}")
    for c in report["comparisons"]:
        if c["metric"] == "rps":
            print(
                f"  {c['candidate']} − {c['reference']}: {c['value']:+.4f} "
                f"[{c['low']:+.4f}, {c['high']:+.4f}] → {c['verdict']}"
            )
    print(f"  {report['caveat']}")


if __name__ == "__main__":
    main()
