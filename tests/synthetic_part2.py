"""A synthetic Part 2 — database and manifest — generated from the column contract.

This repository's only input is Part 2's ``analytics.duckdb``, which is built from
real data and never committed. CI therefore needs a stand-in, and so do the tests
that break things on purpose. This module is that stand-in, used by both:

    uv run python tests/synthetic_part2.py /tmp/part2     # then PART2_DIR=/tmp/part2

Every table the contract lists is created with **exactly the contract's columns and
types**, plus one column the contract does not list (which must not matter). So it
tracks the contract as it grows, and a contract change that the generator was not
told about fails loudly here rather than silently in CI.

The data is one four-club league over three seasons, deterministic, chosen to carry
the situations the feature layer has to get right:

- a **season boundary**, so form carries over and season-to-date resets;
- a **relegated club (4)** and a **promoted club (5)** — club 5 has no history and no
  prior season when it arrives (L7);
- a **match with no xG** in 2025/26, null on both rows (Part 2 trap 7);
- a **partial third season**, which splits.py must treat as live;
- a ``mart_team_season`` row for **every** season, including the one being predicted,
  so the L2 leak (joining the current season's summary) has something to leak.

What it does not try to be is Part 2's own fixture, which carries the Gold-level traps
(awarded and postponed matches, split fixtures). Part 2 handles those before this
repository ever sees a row; the ``part2-integration`` CI job checks the real thing.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import duckdb
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((REPO_ROOT / "contracts" / "part2_columns.json").read_text())
EXPORTED_AT = "2026-09-20T12:00:00+00:00"

# (season, clubs, one date per round). Four clubs → six rounds of a double round
# robin, two matches a round. Rounds are a month apart so walk-forward folds exist.
SEASONS = [
    (
        "2024/25",
        [1, 2, 3, 4],
        ["2024-08-17", "2024-09-14", "2024-10-19", "2024-11-16", "2024-12-14", "2025-01-18"],
    ),
    (
        "2025/26",
        [1, 2, 3, 5],
        ["2025-08-16", "2025-09-13", "2025-10-18", "2025-11-15", "2025-12-13", "2026-01-17"],
    ),
    ("2026/27", [1, 2, 3, 5], ["2026-08-15"]),  # in progress: one round played
]
NO_XG_MATCH = ("2025/26", 2, 0)  # season, round index, match index within the round
STRENGTH = {1: 0.5, 2: 0.2, 3: -0.1, 4: -0.4, 5: -0.3}


def _rounds(clubs: list[int]) -> list[list[tuple[int, int]]]:
    """Double round robin by the circle method: (home, away) pairs per round."""
    n, rotation, first_half = len(clubs), clubs[:], []
    for _ in range(n - 1):
        first_half.append([(rotation[i], rotation[n - 1 - i]) for i in range(n // 2)])
        rotation = [rotation[0], rotation[-1], *rotation[1:-1]]
    return first_half + [[(a, h) for h, a in r] for r in first_half]


def team_matches() -> list[dict]:
    """int_team_matches rows: one per club per match, two per match, mirrored."""
    rng = np.random.default_rng(20260920)
    rows, match_key = [], 0
    for season, clubs, dates in SEASONS:
        for r, (day, fixtures) in enumerate(zip(dates, _rounds(clubs), strict=False)):
            for m, (home, away) in enumerate(fixtures):
                match_key += 1
                xg = {
                    home: 1.3 + STRENGTH[home] - STRENGTH[away] / 2 + rng.uniform(-0.3, 0.3),
                    away: 1.0 + STRENGTH[away] - STRENGTH[home] / 2 + rng.uniform(-0.3, 0.3),
                }
                goals = {club: int(rng.poisson(max(xg[club], 0.1))) for club in (home, away)}
                no_xg = (season, r, m) == NO_XG_MATCH
                for club, opponent, is_home in ((home, away, True), (away, home, False)):
                    gf, ga = goals[club], goals[opponent]
                    xgf, xga = round(xg[club], 2), round(xg[opponent], 2)
                    rows.append(
                        {
                            "match_key": match_key,
                            "match_date": day,
                            "season": season,
                            "competition_code": "PL",
                            "team_key": club,
                            "opponent_key": opponent,
                            "is_home": is_home,
                            "goals_for": gf,
                            "goals_against": ga,
                            "xg_for": None if no_xg else xgf,
                            "xg_against": None if no_xg else xga,
                            "npxg_for": None if no_xg else xgf,
                            "npxg_against": None if no_xg else xga,
                            "xpts": None if no_xg else round(min(max(1.4 + xgf - xga, 0), 3), 2),
                            "points": 3 if gf > ga else 1 if gf == ga else 0,
                        }
                    )
    return rows


def team_seasons(matches: list[dict]) -> list[dict]:
    """mart_team_season rows, aggregated the way Part 2 does, for the columns read."""
    by_team: dict[tuple, list[dict]] = defaultdict(list)
    for row in matches:
        by_team[(row["team_key"], row["season"], row["competition_code"])].append(row)
    scheduled = {season: 2 * (len(clubs) - 1) for season, clubs, _ in SEASONS}
    played_by_season: dict[str, list[int]] = defaultdict(list)
    for (_, season, _), rows in by_team.items():
        played_by_season[season].append(len(rows))

    out = []
    for (team, season, competition), rows in sorted(by_team.items()):
        with_xg = [r for r in rows if r["npxg_for"] is not None]
        out.append(
            {
                "team_key": team,
                "season": season,
                "competition_code": competition,
                "played": len(rows),
                "points_per_match": sum(r["points"] for r in rows) / len(rows),
                "matches_with_xg": len(with_xg),
                "npxg_for": sum(r["npxg_for"] for r in with_xg) if with_xg else None,
                "npxg_against": sum(r["npxg_against"] for r in with_xg) if with_xg else None,
                "is_season_complete": min(played_by_season[season]) == scheduled[season],
                "last_match_date": max(r["match_date"] for r in rows),
            }
        )
    return out


ROWS = {"int_team_matches": team_matches()}
ROWS["mart_team_season"] = team_seasons(ROWS["int_team_matches"])
MATCHES = len({r["match_key"] for r in ROWS["int_team_matches"]})


def make_part2(
    root: Path,
    *,
    matches_in_manifest: int = MATCHES,
    drop_column: str | None = None,
    retype: dict[str, str] | None = None,
    drop_table: str | None = None,
) -> Path:
    """Write a Part 2 checkout — ``data/analytics.duckdb`` and its manifest — under ``root``.

    ``drop_column`` and ``retype`` apply to ``int_team_matches``; each keyword
    introduces one defect, for the tests that check the build refuses it.
    """
    missing = set(CONTRACT["tables"]) - set(ROWS) - {"stg_matches"}
    if missing:
        raise RuntimeError(
            f"The contract lists {sorted(missing)}, which tests/synthetic_part2.py does not "
            "generate. Add rows for it here."
        )

    (root / "data" / "gold").mkdir(parents=True, exist_ok=True)
    db = root / "data" / "analytics.duckdb"
    db.unlink(missing_ok=True)
    con = duckdb.connect(str(db))

    con.execute(f"CREATE TABLE stg_matches AS SELECT range AS match_key FROM range({MATCHES})")

    for table, rows in ROWS.items():
        columns = [(c["name"], c["type"]) for c in CONTRACT["tables"][table]["columns"]]
        if table == "int_team_matches":
            columns = [
                (name, (retype or {}).get(name, type_))
                for name, type_ in columns
                if name != drop_column
            ]
        con.execute(
            f"CREATE TABLE {table} ("
            + ", ".join(f"{name} {type_}" for name, type_ in columns)
            + ")"
        )
        names = [name for name, _ in columns]
        con.executemany(
            f"INSERT INTO {table} VALUES ({', '.join('?' * len(names))})",
            [[row[n] for n in names] for row in rows],
        )
        # A column Part 2 has and this repository does not read must not matter.
        con.execute(f"ALTER TABLE {table} ADD COLUMN something_new_in_part2 INTEGER")

    if drop_table:
        con.execute(f"DROP TABLE {drop_table}")
    con.close()

    dates = [date.fromisoformat(r["match_date"]) for r in ROWS["int_team_matches"]]
    manifest = {
        "exported_at": EXPORTED_AT,
        "source": "synthetic Part 2 (tests/synthetic_part2.py)",
        "match_date_range": {"min": str(min(dates)), "max": str(max(dates))},
        "tables": {
            "fact_matches": {"rows": matches_in_manifest},
            "fact_player_events": {"rows": 0},
            "fact_shots": {"rows": 0},
        },
    }
    (root / "data" / "gold" / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return db


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python tests/synthetic_part2.py <directory>")
    target = Path(sys.argv[1])
    make_part2(target)
    print(f"synthetic Part 2 written to {target}: {MATCHES} matches")
