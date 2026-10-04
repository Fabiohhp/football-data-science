"""Every chart draws from the committed reports, and the README's images exist.

The charts read ``reports/*.json`` and nothing else, so this runs anywhere — no data,
no database — which is also why CI can execute the notebooks.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from football_ml import viz  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
MATCH = json.loads((REPO_ROOT / "reports" / "match_outcome.json").read_text())
SHOTS = json.loads((REPO_ROOT / "reports" / "shot_xg.json").read_text())

CHARTS = {
    "split": lambda: viz.split_timeline(MATCH["plan"]),
    "comparisons": lambda: viz.interval_plot(
        MATCH["holdout"]["comparisons"], [("dc_xg", "elo"), ("elo", "base_rates")],
        metric="rps", unit="RPS",
    ),
    "calibration": lambda: viz.reliability(MATCH["holdout"]["calibration"], ["dc_xg", "elo"]),
    "shot calibration": lambda: viz.binary_reliability(
        SHOTS["holdout"]["calibration"], ["model", "understat"], labels={"model": "This model"}
    ),
    "situations": lambda: viz.situation_dumbbell(SHOTS["holdout"]["by_situation"]),
    "xg map": lambda: viz.xg_surface(SHOTS["surface"]),
}  # fmt: skip


@pytest.mark.parametrize("name", CHARTS)
def test_every_chart_draws_from_the_committed_reports(name):
    viz.apply_style()
    fig = CHARTS[name]()
    viz.title_block(fig, "title", "subtitle")
    viz.annotate_source(fig, MATCH["citation"])
    fig.canvas.draw()
    assert fig.axes
    plt.close(fig)


def test_labels_name_the_shot_model_not_the_match_model():
    fig = viz.interval_plot(
        SHOTS["holdout"]["comparisons"], [("model", "baseline")], metric="log_loss",
        unit="log loss", labels={"model": "This model"},
    )  # fmt: skip
    assert fig.axes[0].get_yticklabels()[0].get_text() == "This model  vs  Distance + angle"
    plt.close(fig)


def test_every_image_the_readme_shows_exists():
    readme = (REPO_ROOT / "README.md").read_text()
    images = re.findall(r"\]\((docs/images/[^)]+\.png)\)", readme)
    assert images, "the README shows no charts"
    for image in images:
        assert (REPO_ROOT / image).exists(), f"{image} is referenced but missing"
