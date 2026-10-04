"""Charts for the notebooks and the README, in Part 2's style (CLAUDE.md §3, §10).

Every chart here draws a block of a committed ``reports/*.json`` — nothing is fitted
or scored in this module, so a notebook that uses it runs anywhere, with no data, and
shows exactly what the pipeline reported.

**The palette is Part 2's**, so the portfolio reads as one piece: an off-white surface
(GitHub's notebook view shows a transparent background as dark), recessive hairline
grids, and two hues, validated together for colour-blind separation. The rule for
which hue goes where is the same on every chart:

- **blue** — the model the chart is about;
- **orange** — what it is measured against (Elo, or Understat);
- **grey** — reference: a baseline, a diagonal, an excluded period.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.figure import Figure

# --- palette (Part 2's src/football_analytics/viz.py) ------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
MODEL = "#2a78d6"  # Part 2's ACCENT
MODEL_SOFT = "#a8c9ee"
AGAINST = "#eb6834"  # Part 2's GOAL hue
REFERENCE = "#9a9893"

OUTCOME_NAMES = {"H": "Home win", "D": "Draw", "A": "Away win"}
METHOD_NAMES = {
    "base_rates": "League base rates",
    "elo": "Elo rating",
    "model": "Form model (#1)",
    "dc_goals": "Dixon-Coles, goals",
    "dc_xg": "Dixon-Coles, xG",
    "baseline": "Distance + angle",
    "understat": "Understat, as published",
    "understat_recalibrated": "Understat, recalibrated",
}


def apply_style() -> None:
    """The rcParams every chart inherits — Part 2's, unchanged."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "font.size": 10,
            "text.color": INK,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": INK_MUTED,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
        }
    )


def title_block(fig: Figure, title: str, subtitle: str | None = None) -> None:
    """A bold left-aligned figure title with a plain-language subtitle beneath.

    Placed in inches from the top, not as a fraction of the figure: a fraction puts
    the subtitle on top of the title in a short figure and far below it in a tall one.
    """
    height = fig.get_figheight()
    fig.text(0.01, 1 - 0.06 / height, title, fontsize=12, fontweight="bold", va="top")
    if subtitle:
        fig.text(0.01, 1 - 0.33 / height, subtitle, fontsize=9.5, color=INK_MUTED, va="top")


def annotate_source(fig: Figure, citation: str) -> None:
    """The snapshot every chart was drawn from, under it (CLAUDE.md §7)."""
    fig.text(0.01, 0.005, citation, fontsize=7.5, color=INK_MUTED, va="bottom")


# --- the split ---------------------------------------------------------------------


def _month_index(label: str, origin: date) -> int:
    year, month = map(int, label.split("-"))
    return (year - origin.year) * 12 + (month - origin.month)


def split_timeline(plan: dict, *, first_month: str = "2024-08") -> Figure:
    """Every walk-forward fold as a row: what it trained on, and the month it predicted.

    Drawn from the report's own ``plan`` block, so the picture is the split that
    produced the scores, not a description of one.
    """
    origin = date(*map(int, first_month.split("-")), 1)
    folds = [("validation", f) for f in plan["validation_folds"]] + [
        ("holdout", f) for f in plan["holdout_folds"]
    ]
    last = max(_month_index(f["month"], origin) for _, f in folds)
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    fig.subplots_adjust(top=0.84, bottom=0.17, left=0.13, right=0.98)

    for row, (period, fold) in enumerate(folds):
        m = _month_index(fold["month"], origin)
        y = len(folds) - 1 - row
        ax.add_patch(Rectangle((0, y - 0.32), m, 0.64, color=GRID, lw=0))
        colour = MODEL if period == "validation" else AGAINST
        ax.add_patch(Rectangle((m + 0.06, y - 0.32), 0.88, 0.64, color=colour, lw=0))
        ax.text(
            last + 1.35,
            y,
            f"{fold['test']:,}",
            va="center",
            ha="left",
            fontsize=8,
            color=INK_MUTED,
        )

    ax.text(
        last + 1.35,
        len(folds) + 0.15,
        "matches\npredicted",
        fontsize=8,
        color=INK_MUTED,
        va="bottom",
    )

    ticks = [m for m in range(0, last + 1) if (origin.month - 1 + m) % 12 in (7, 0, 4)]
    ax.set_xticks([t + 0.5 for t in ticks])
    ax.set_xticklabels(
        [
            date(
                origin.year + (origin.month - 1 + t) // 12, (origin.month - 1 + t) % 12 + 1, 1
            ).strftime("%b\n%Y")
            for t in ticks
        ]
    )
    ax.set_yticks(range(len(folds)))
    ax.set_yticklabels([f["month"] for _, f in reversed(folds)], fontsize=8)
    ax.set_xlim(0, last + 3.2)
    ax.set_ylim(-0.8, len(folds) - 0.2)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    n_val = len(plan["validation_folds"])
    divider = len(folds) - n_val - 0.5
    ax.axhline(divider, color=INK_MUTED, lw=0.8)
    # The space to the right of the validation blocks is empty: both labels go there,
    # clear of every bar.
    free_x = max(_month_index(f["month"], origin) for f in plan["validation_folds"]) + 1.8
    ax.text(
        free_x,
        len(folds) - 2,
        "↑ validation: every setting and model\n   choice was made on these months",
        fontsize=8.5,
        color=MODEL,
        va="center",
    )
    ax.text(
        free_x,
        divider + 1.2,
        f"↓ hold-out {plan['holdout_season']}: scored once,\n   settings already fixed",
        fontsize=8.5,
        color=AGAINST,
        va="center",
    )

    from matplotlib.lines import Line2D

    ax.legend(
        handles=[
            Line2D([], [], color=GRID, lw=8, label="trained on (everything before)"),
            Line2D([], [], color=MODEL, lw=8, label="validation month predicted"),
            Line2D([], [], color=AGAINST, lw=8, label="hold-out month predicted"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        fontsize=8.5,
    )
    title_block(
        fig,
        "Walk-forward: every fold trains on the past and predicts the next month",
        f"{len(folds)} folds. The first three months are burn-in; "
        f"{', '.join(plan['excluded_live_seasons'])} is in progress and used for nothing.",
    )
    return fig


# --- comparisons with intervals ------------------------------------------------------


def interval_plot(
    comparisons: list[dict],
    pairs: list[tuple[str, str]],
    *,
    metric: str,
    unit: str,
    labels: dict[str, str] | None = None,
) -> Figure:
    """Paired differences with their bootstrap intervals, one row per comparison.

    A filled marker is an interval that excludes zero — one model measurably better;
    a hollow one crosses zero and the two cannot be told apart. Left of zero means
    the first-named method is better (every metric here is lower-is-better).
    """
    names = METHOD_NAMES | (labels or {})
    rows = []
    for a, b in pairs:
        c = next(
            c
            for c in comparisons
            if (c["candidate"], c["reference"], c["metric"]) == (a, b, metric)
        )
        rows.append((f"{names[a]}  vs  {names[b]}", c))

    fig, ax = plt.subplots(figsize=(9.2, 0.62 * len(rows) + 1.9))
    height = 0.62 * len(rows) + 1.9
    fig.subplots_adjust(top=1 - 0.75 / height, bottom=1.05 / height, left=0.36, right=0.8)
    for i, (_label, c) in enumerate(rows):
        y = len(rows) - 1 - i
        excludes = c["high"] < 0 or c["low"] > 0
        ax.plot([c["low"], c["high"]], [y, y], color=MODEL, lw=2, solid_capstyle="round")
        ax.plot(
            c["value"],
            y,
            "o",
            ms=8,
            color=MODEL,
            markerfacecolor=MODEL if excludes else SURFACE,
            markeredgewidth=2,
        )
        verdict = (
            "cannot be told apart"
            if not excludes
            else ("first is better" if c["high"] < 0 else "second is better")
        )
        ax.text(
            1.02,
            y,
            verdict,
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=9,
            color=INK if excludes else INK_MUTED,
        )
    ax.axvline(0, color=INK, lw=0.9)
    # Few ticks: these differences are small numbers with long labels, and a default
    # locator runs them into each other.
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, symmetric=False))
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=9)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(
        f"difference in {unit} (first − second), 95% paired bootstrap interval\n"
        "← first is better        second is better →"
    )
    return fig


# --- calibration -----------------------------------------------------------------------


def reliability(
    calibration: dict[str, list[dict]], methods: list[str], *, min_matches: int = 20
) -> Figure:
    """Predicted against observed, per outcome: on the diagonal is calibrated.

    One panel per outcome — a model can be calibrated overall and still wrong about
    draws (L11). Bins with fewer than ``min_matches`` matches are left out, not merged.
    """
    colours = dict(zip(methods, [MODEL, AGAINST], strict=False))
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.9), sharey=True)
    fig.subplots_adjust(top=0.78, bottom=0.27, left=0.07, right=0.99, wspace=0.12)
    for ax, outcome in zip(axes, ["H", "D", "A"], strict=True):
        ax.plot([0, 0.9], [0, 0.9], color=REFERENCE, lw=1)
        for method in methods:
            rows = [
                r for r in calibration[method] if r["outcome"] == outcome and r["n"] >= min_matches
            ]
            x = [r["mean_predicted"] for r in rows]
            y = [r["observed_rate"] for r in rows]
            ax.plot(
                x,
                y,
                "-o",
                color=colours[method],
                lw=1.6,
                ms=5.5,
                markeredgecolor=SURFACE,
                markeredgewidth=1.2,
                label=METHOD_NAMES[method],
            )
        ax.set_title(OUTCOME_NAMES[outcome], fontsize=10)
        ax.set_xlim(0, 0.9)
        ax.set_ylim(0, 0.9)
        ax.set_aspect("equal")
        ax.set_xticks([0, 0.25, 0.5, 0.75])
        ax.set_xticklabels(["0%", "25%", "50%", "75%"])
        ax.set_xlabel("model said")
    axes[0].set_yticks([0, 0.25, 0.5, 0.75])
    axes[0].set_yticklabels(["0%", "25%", "50%", "75%"])
    axes[0].set_ylabel("happened")
    axes[0].legend(loc="upper left", fontsize=8.5)
    return fig


def binary_reliability(
    calibration: dict[str, list[dict]],
    methods: list[str],
    *,
    min_shots: int = 100,
    labels: dict[str, str] | None = None,
) -> Figure:
    """Predicted xG against the share that went in, per probability bin."""
    names = METHOD_NAMES | (labels or {})
    colours = dict(zip(methods, [MODEL, AGAINST, REFERENCE], strict=False))
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    fig.subplots_adjust(top=0.84, bottom=0.14, left=0.12, right=0.97)
    ax.plot([0, 0.85], [0, 0.85], color=REFERENCE, lw=1)
    for method in methods:
        rows = [r for r in calibration[method] if r["n"] >= min_shots]
        ax.plot(
            [r["mean_predicted"] for r in rows],
            [r["observed_rate"] for r in rows],
            "-o",
            color=colours[method],
            lw=1.6,
            ms=5.5,
            markeredgecolor=SURFACE,
            markeredgewidth=1.2,
            label=names[method],
        )
    ax.set_xlim(0, 0.85)
    ax.set_ylim(0, 0.85)
    ax.set_aspect("equal")
    ax.set_xlabel("predicted chance of a goal")
    ax.set_ylabel("share that went in")
    ax.legend(loc="upper left", fontsize=8.5)
    return fig


# --- shots ------------------------------------------------------------------------------


def situation_dumbbell(by_situation: dict[str, dict]) -> Figure:
    """AUC by situation, the model against Understat: where the gap is, and where not."""
    order = sorted(by_situation, key=lambda s: -by_situation[s]["model"]["n"])
    names = {
        "OpenPlay": "Open play",
        "FromCorner": "From a corner",
        "SetPiece": "Other set piece",
        "DirectFreekick": "Direct free kick",
    }
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    fig.subplots_adjust(top=0.78, bottom=0.2, left=0.2, right=0.84)
    for i, s in enumerate(order):
        y = len(order) - 1 - i
        m = by_situation[s]["model"]["auc"]
        u = by_situation[s]["understat"]["auc"]
        ax.plot([m, u], [y, y], color=GRID, lw=3, solid_capstyle="round", zorder=1)
        ax.plot(
            m, y, "o", ms=9, color=MODEL, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3
        )
        ax.plot(
            u, y, "o", ms=9, color=AGAINST, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3
        )
        ax.text(
            1.02,
            y,
            f"{by_situation[s]['model']['n']:,} shots",
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=8.5,
            color=INK_MUTED,
        )
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names.get(s, s) for s in reversed(order)])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("AUC — how well goals are told from misses (0.5 = a coin toss)")
    from matplotlib.lines import Line2D

    ax.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color=MODEL, ms=8, label="this model"),
            Line2D([], [], marker="o", ls="", color=AGAINST, ms=8, label="Understat"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        fontsize=8.5,
    )
    return fig


def xg_surface(surface: dict) -> Figure:
    """What the shot model learned: its xG over the attacking third, for two shots.

    The final model's predictions on a grid (``reports/shot_xg.json``, ``surface``),
    drawn as zones on an attacking-half pitch in Understat's orientation (Part 2
    trap 10). One sequential hue, with the zone edges at round xG values.
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from mplsoccer import VerticalPitch

    levels = [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    cmap = ListedColormap(mpl.colormaps["Blues"](np.linspace(0.08, 0.92, len(levels) - 1)))
    norm = BoundaryNorm(levels, cmap.N)
    xs = np.array(surface["x"]) * 100
    ys = np.array(surface["y"]) * 100
    gx, gy = np.meshgrid(xs, ys, indexing="ij")

    pitch = VerticalPitch(
        pitch_type="opta",
        half=True,
        pitch_color=SURFACE,
        line_color="#8f8d88",
        linewidth=1.0,
        pad_bottom=-18,
        line_zorder=3,
    )
    fig, axes = pitch.draw(nrows=1, ncols=2, figsize=(9.4, 4.4))
    fig.set_facecolor(SURFACE)
    fig.subplots_adjust(top=0.86, bottom=0.2, left=0.01, right=0.99, wspace=0.04)
    mappable = None
    for ax, (label, values) in zip(axes, surface["profiles"].items(), strict=True):
        # A vertical pitch draws pitch (x, y) at axes (y, x) — checked against
        # pitch.scatter — so the grid goes in swapped; the axis limits carry the flip.
        mappable = ax.contourf(
            gy, gx, np.array(values), levels=levels, cmap=cmap, norm=norm, zorder=1
        )
        ax.set_title(label, fontsize=10, loc="left", pad=4)
    bar = fig.colorbar(
        mappable,
        cax=fig.add_axes((0.22, 0.115, 0.56, 0.03)),
        orientation="horizontal",
        ticks=levels,
    )
    bar.ax.xaxis.set_label_position("top")
    bar.ax.set_xticklabels([f"{v:g}" for v in levels], fontsize=8.5)
    bar.set_label("expected goals (chance the shot is scored)", color=INK_MUTED, fontsize=9)
    bar.outline.set_visible(False)
    return fig
