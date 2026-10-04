"""Predictions, ready to display: models + reports → exports/predictions/*.json.

    uv run python scripts/export_predictions.py

Part 2's dashboard shows predictions "as pre-computed JSON the site reads, never a
model running live" (its §9; CLAUDE.md §9 here). This script writes that JSON. Like
Part 2's own export it **selects and formats, never computes**: every probability
was written by training or by prospective scoring, every score and interval comes
from ``reports/``, and the only operations here are choosing columns, joining club
names and rounding. A ``groupby`` here would be a missing report.

What it shows:

- **Dixon-Coles on xG** as the model, because it had the best *validation* score of
  the five methods (CLAUDE.md §12, phase 3) — the only fair way to choose what to
  display. Elo beside it, as the baseline it could not be told apart from.
- The **2025/26 hold-out**, every match walked forward month by month: what the model
  said before each match, and what happened.
- The **prospective** matches, once a frozen model has any: played after it was
  frozen, scored by ``football_ml.prospective``.

Written to ``exports/``, gitignored until CLAUDE.md §13 decides on publishing. Part 2
copies the files it shows from here; nothing in this repository writes into Part 2.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from football_ml import data, freeze, predictions  # noqa: E402
from football_ml.evaluate import OUTCOMES  # noqa: E402

EXPORT_DIR = REPO_ROOT / "exports" / "predictions"
DISPLAY = "dc_xg"
BASELINE = "elo"
LABELS = {
    "dc_xg": "Dixon-Coles on xG",
    "dc_goals": "Dixon-Coles on goals",
    "elo": "Elo rating",
    "model": "form model",
    "base_rates": "league base rates",
}
LEAGUES = {
    "PL": "Premier League",
    "PD": "La Liga",
    "BL1": "Bundesliga",
    "SA": "Serie A",
    "FL1": "Ligue 1",
}


def _r(x, places=3):
    return None if x is None else round(float(x), places)


def _scores(block: dict) -> dict:
    return {
        m: {k: _r(v, 4) for k, v in s.items() if k != "n"} | {"n": s["n"]} for m, s in block.items()
    }


def _comparison(
    comparisons: list[dict], candidate: str, reference: str, metric="rps"
) -> dict | None:
    for c in comparisons:
        if (c["candidate"], c["reference"], c["metric"]) == (candidate, reference, metric):
            return {
                "candidate": LABELS[candidate],
                "reference": LABELS[reference],
                "metric": metric,
                "difference": _r(c["value"], 4),
                "low": _r(c["low"], 4),
                "high": _r(c["high"], 4),
                "verdict": c["verdict"]
                .replace(candidate, LABELS[candidate])
                .replace(reference, LABELS[reference]),
                "matches": c["n"],
            }
    return None


def _calibration(rows: list[dict]) -> list[dict]:
    return [
        {
            "outcome": r["outcome"],
            "bin": f"{r['bin_low']:.1f}–{r['bin_high']:.1f}",
            "matches": r["n"],
            "predicted": _r(r["mean_predicted"]),
            "observed": _r(r["observed_rate"]),
        }
        for r in rows
    ]


def _rows(frame, names: dict[int, str]) -> list[dict]:
    """One display row per match. Selection only: the probability given to what
    happened is the column named by the result, looked up, not derived."""
    out = []
    for m in frame.sort_values(["as_of_date", "match_key"]).itertuples(index=False):
        row = m._asdict()
        out.append(
            {
                "date": str(m.as_of_date.date()),
                "league": m.competition_code,
                "home": names.get(m.home_team_key, f"club {m.home_team_key}"),
                "away": names.get(m.away_team_key, f"club {m.away_team_key}"),
                "score": f"{m.target_home_goals}–{m.target_away_goals}",
                "result": m.result,
                "model": [_r(row[f"p_{DISPLAY}_{o}"]) for o in OUTCOMES],
                "baseline": [_r(row[f"p_{BASELINE}_{o}"]) for o in OUTCOMES],
                "model_on_result": _r(row[f"p_{DISPLAY}_{m.result}"]),
            }
        )
    return out


def _write(name: str, payload) -> Path:
    path = EXPORT_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    return path


def main() -> None:
    features = data.open_features()
    names = dict(features.con.execute("SELECT team_key, team_name FROM src_teams").fetchall())
    snapshot = features.snapshot
    features.con.close()

    report_path = REPO_ROOT / "reports" / "match_outcome.json"
    holdout_rows = predictions.read("match_outcome_holdout")
    if holdout_rows is None or not report_path.exists():
        raise SystemExit(
            "No hold-out predictions yet. Train first:\n"
            "  uv run python -m football_ml.train match_outcome"
        )
    report = json.loads(report_path.read_text())
    h = report["holdout"]

    card_path = freeze.latest()
    prospective = {"status": "not frozen", "reason": "no model has been frozen yet"}
    prospective_rows = []
    if card_path is not None:
        card = json.loads(card_path.read_text())
        freeze_info = {
            "freeze_id": card["freeze_id"],
            "frozen_at": card["frozen_at"],
            "scores_matches_after": card["scores_matches_after"],
            "trained_on": card["training_range"],
            "frozen_on": card["citation"],
            "commit": card["git"]["commit"],
        }
        report_p = REPO_ROOT / "reports" / f"prospective_{card['freeze_id']}.json"
        rows_p = predictions.read(f"prospective_{card['freeze_id']}")
        if report_p.exists() and rows_p is not None:
            pr = json.loads(report_p.read_text())
            prospective = freeze_info | {
                "status": "scored",
                "scored_on": pr["scored_on"],
                "matches": pr["matches"],
                "date_range": pr["date_range"],
                "scores": _scores({m: pr["scores"][m] for m in (DISPLAY, BASELINE, "base_rates")}),
                "comparison": _comparison(pr["comparisons"], DISPLAY, BASELINE),
                "caveat": pr["caveat"],
            }
            prospective_rows = _rows(rows_p, names)
        else:
            prospective = freeze_info | {
                "status": "waiting",
                "reason": "frozen, and waiting for matches played after "
                f"{card['scores_matches_after']} to reach a newer snapshot",
            }

    meta = {
        "snapshot": {
            "exported_at": snapshot["exported_at"],
            "match_date_min": snapshot["match_date_min"],
            "match_date_max": snapshot["match_date_max"],
        },
        "model": {
            "key": DISPLAY,
            "label": LABELS[DISPLAY],
            "settings": report["selected"][DISPLAY],
        },
        "baseline": {"key": BASELINE, "label": LABELS[BASELINE]},
        "why_this_model": "best ranked probability score on the validation season (2024/25) "
        "of the five methods; the hold-out did not choose it",
        "outcomes": list(OUTCOMES),
        "leagues": LEAGUES,
        "holdout": {
            "season": h["season"],
            "matches": h["n"],
            "scores": _scores({m: h["scores"][m] for m in LABELS}),
            "labels": LABELS,
            "comparison": _comparison(h["comparisons"], DISPLAY, BASELINE),
            "calibration": {
                DISPLAY: _calibration(h["calibration"][DISPLAY]),
                BASELINE: _calibration(h["calibration"][BASELINE]),
            },
            "draws": {m: {k: _r(v) for k, v in h["draws"][m].items()} for m in (DISPLAY, BASELINE)},
        },
        "prospective": prospective,
        "citation": report["citation"],
        "no_market": "Compared with base rates and Elo only — Gold holds no betting odds.",
    }

    written = [_write("meta.json", meta)]
    by_league = holdout_rows.groupby("competition_code", sort=True)  # a split, not a sum
    for league, rows in by_league:
        written.append(_write(f"holdout_{league}.json", _rows(rows, names)))
    written.append(_write("prospective.json", prospective_rows))
    for path in written:
        print(f"wrote {path.relative_to(REPO_ROOT)} ({path.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
