"""The leakage test has teeth (CLAUDE.md §6, §12 phase 1).

``assert_no_feature_reads_the_future`` is the test this repository rests on, and a
test never seen failing proves nothing. Each case here copies the dbt project,
introduces one specific leak into the feature SQL, builds it against the synthetic
Part 2 and asserts that the leakage test — by name — is what fails.

The leaks are the two a reasonable person could write by accident:

- **L1** — the rolling window ending at ``current row``, as Part 2's chart does, so
  a match's own points are in its own form;
- **L2** — joining last season's summary on the *current* season, so a mid-season
  row holds the whole season's result.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from conftest import copy_project, make_part2

FORM_WINDOW = "rows between {{ var('form_window') }} preceding and 1 preceding"
PRIOR_JOIN = "prior.applies_to_season = w.season"


def _leak(project: Path, old: str, new: str) -> None:
    sql = project / "models" / "features" / "feat_team_form.sql"
    text = sql.read_text()
    assert text.count(old) == 1, f"expected exactly one {old!r} in {sql.name} — did it change?"
    sql.write_text(text.replace(old, new))


def test_the_unmodified_features_pass(tmp_path, dbt_build):
    make_part2(tmp_path / "part2")

    result = dbt_build()

    assert result.success, result.log
    for model in ("feat_team_form", "feat_match_outcome"):
        assert f"PASS assert_no_feature_reads_the_future_{model}" in result.log


@pytest.mark.parametrize(
    ("leak", "old", "new"),
    [
        (
            "L1: form window includes the match itself",
            FORM_WINDOW,
            FORM_WINDOW.replace("1 preceding", "current row"),
        ),
        (
            "L2: season summary joined on the current season",
            PRIOR_JOIN,
            "prior.summarised_season = w.season",
        ),
    ],
)
def test_the_leakage_test_catches(tmp_path, dbt_build, leak, old, new):
    make_part2(tmp_path / "part2")
    project = copy_project(tmp_path / "dbt")
    _leak(project, old, new)

    result = dbt_build(project)

    assert not result.success, f"{leak}: the build passed"
    # The leakage test itself must fail, on the model the leak is in — not merely the
    # build, which another test can fail while this one is skipped (it happened: see
    # dbt/macros/test_assert_no_feature_reads_the_future.sql).
    caught = re.search(r"FAIL (\d+) assert_no_feature_reads_the_future_feat_team_form", result.log)
    assert caught, f"{leak}: the build failed, but not on the leakage test:\n{result.log}"


def test_a_feature_model_without_the_leakage_test_fails_the_build(tmp_path, dbt_build):
    """A new feat_ model that forgets the test would leak unobserved; the build says so."""
    make_part2(tmp_path / "part2")
    project = copy_project(tmp_path / "dbt")
    (project / "models" / "features" / "feat_forgotten.sql").write_text(
        "select match_key, as_of_date, max_input_date from {{ ref('feat_match_outcome') }}\n"
    )

    result = dbt_build(project)

    assert not result.success
    assert re.search(r"FAIL 1 assert_every_feature_model_is_leakage_tested", result.log), result.log
