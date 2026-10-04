"""Collect the charts the README shows from the executed notebooks.

    uv run python scripts/export_readme_images.py

The README's figures are lifted from the notebooks' committed outputs, byte for byte —
never redrawn by a second copy of the plotting code. A chart in the README that had
drifted from the analysis it illustrates would be worse than no chart. Part 2 does the
same (its scripts/export_readme_images.py).

A figure is chosen by a cell tag, ``readme:<file name>``, rather than by cell number,
so adding a cell to a notebook cannot silently swap one chart for another. A tag whose
cell has no image — the notebook was not executed — is an error, not a skip.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import nbformat

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPO_ROOT / "notebooks"
IMAGES = REPO_ROOT / "docs" / "images"
TAG = "readme:"


def main() -> int:
    IMAGES.mkdir(parents=True, exist_ok=True)
    written, missing = [], []
    for path in sorted(NOTEBOOKS.glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        for cell in notebook.cells:
            for tag in cell.get("metadata", {}).get("tags", []):
                if not tag.startswith(TAG):
                    continue
                name = tag.removeprefix(TAG)
                pngs = [
                    o["data"]["image/png"]
                    for o in cell.get("outputs", [])
                    if "image/png" in o.get("data", {})
                ]
                if len(pngs) != 1:
                    missing.append(f"{path.name}: {name} ({len(pngs)} images in its cell)")
                    continue
                (IMAGES / name).write_bytes(base64.b64decode(pngs[0]))
                written.append(name)
    for name in written:
        print(f"wrote docs/images/{name}")
    if missing:
        print("not exported — execute the notebooks first:", *missing, sep="\n  ", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
