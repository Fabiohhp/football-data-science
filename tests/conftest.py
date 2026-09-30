"""Running the real ``dbt build`` against a synthetic Part 2.

Several test files break something on purpose — Part 2's schema, or this
repository's own feature SQL — and assert the build notices. They share this
fixture so each test states only the one thing it breaks. The synthetic Part 2
itself is ``tests/synthetic_part2.py``, which CI also uses.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from dbt.cli.main import dbtRunner

from synthetic_part2 import CONTRACT, MATCHES, REPO_ROOT, make_part2

__all__ = ["CONTRACT", "MATCHES", "REPO_ROOT", "BuildResult", "copy_project", "make_part2"]


@dataclass
class BuildResult:
    success: bool
    log: str


@pytest.fixture
def dbt_build(tmp_path, monkeypatch):
    """Run the real ``dbt build`` against the synthetic Part 2 under ``tmp_path/part2``.

    Everything dbt writes — features.duckdb, target/, logs — goes to the temp
    directory, so the tests never touch data/ or the developer's own build. Pass
    ``project_dir`` to build a modified copy of the project (see :func:`copy_project`).
    """
    monkeypatch.chdir(REPO_ROOT)  # the contract path is relative to the repo root
    monkeypatch.setenv("PART2_DIR", str(tmp_path / "part2"))
    monkeypatch.setenv("FEATURES_DUCKDB", str(tmp_path / "features.duckdb"))
    monkeypatch.delenv("GOLD_SNAPSHOT_DIR", raising=False)

    def run(project_dir: Path = REPO_ROOT / "dbt", *, in_subprocess: bool = False) -> BuildResult:
        args = [
            "build",
            "--project-dir", str(project_dir),
            "--profiles-dir", str(REPO_ROOT / "dbt"),
            "--target-path", str(tmp_path / "target"),
            "--log-path", str(tmp_path / "logs"),
            "--no-partial-parse",
        ]  # fmt: skip
        if in_subprocess:
            # dbt-duckdb keeps its connection to features.duckdb open for the life of
            # the process, and DuckDB refuses a second, read-only one beside it. A test
            # that opens the result the way the loader does builds in its own process.
            dbt = Path(sys.executable).parent / "dbt"
            proc = subprocess.run([str(dbt), *args], capture_output=True, text=True)
            return BuildResult(success=proc.returncode == 0, log=proc.stdout + proc.stderr)

        messages: list[str] = []
        result = dbtRunner(callbacks=[lambda event: messages.append(event.info.msg)]).invoke(args)
        log = "\n".join(messages)
        if result.exception is not None:
            log += f"\n{result.exception}"
        return BuildResult(success=result.success, log=log)

    return run


def copy_project(destination: Path) -> Path:
    """A copy of dbt/ without its build output, for a test to modify."""
    return Path(
        shutil.copytree(
            REPO_ROOT / "dbt",
            destination,
            ignore=shutil.ignore_patterns("target", "logs", ".user.yml"),
        )
    )
