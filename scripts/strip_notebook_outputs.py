"""Remove execution outputs from the public workflow notebooks.

Clearing outputs avoids leaking local filesystem paths and keeps notebooks
focused on executable methodology. Notebook source cells are preserved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOKS = [
    PROJECT_ROOT / "notebooks" / name
    for name in (
        "01_data_understanding.ipynb",
        "02_data_preparation.ipynb",
        "03_modeling_and_tradeoff.ipynb",
        "04_privacy_analysis.ipynb",
        "05_results_summary.ipynb",
    )
]


def strip_outputs(path: Path) -> tuple[int, int]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    output_count = 0
    executed_count = 0
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        output_count += len(cell.get("outputs", []))
        executed_count += int(cell.get("execution_count") is not None)
        cell["outputs"] = []
        cell["execution_count"] = None
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_count, executed_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=Path, action="append", dest="notebooks")
    args = parser.parse_args()
    notebooks = args.notebooks or DEFAULT_NOTEBOOKS
    for notebook in notebooks:
        path = notebook if notebook.is_absolute() else PROJECT_ROOT / notebook
        outputs, executions = strip_outputs(path)
        print(f"{path}: cleared {outputs} outputs and {executions} execution counts")


if __name__ == "__main__":
    main()
