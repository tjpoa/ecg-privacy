"""Audit the Git working tree for common public-release problems.

This script is intentionally dependency-free so it can be run before the
scientific Python environment is installed.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PREFIXES = (
    "data/raw/",
    "data/interim/",
    "data/processed/",
    "outputs/",
    "output/",
    "tmp/",
    "venv/",
    ".venv/",
    "manuscript/",
    "notebooks/",
)
REQUIRED_FILES = (
    "README.md",
    ".gitignore",
    "requirements.txt",
    "THIRD_PARTY_NOTICES.md",
    "configs/experiment_config.yaml",
    "data/README.md",
    "docs/final_protocol.md",
    "docs/reproduction.md",
    "figures/README.md",
    "figures/main_utility_linkability_tradeoff.png",
    "reproducibility/README.md",
    "reproducibility/tables/main_utility_linkability_tradeoff_figure_data.csv",
    "reproducibility/tables/main_paired_uncertainty_summary.csv",
    "reproducibility/tables/article_canonical_representation_table.csv",
    "scripts/export_public_aggregate_tables.py",
    "scripts/verify_public_release.py",
    "tests/test_group_splits.py",
)
FORBIDDEN_DATA_SUFFIXES = {
    ".hea",
    ".mat",
    ".npy",
    ".npz",
    ".parquet",
    ".pkl",
    ".joblib",
    ".h5",
    ".hdf5",
    ".pt",
    ".pth",
    ".onnx",
}
FORBIDDEN_PUBLIC_SUFFIXES = {".tex", ".bib", ".bst", ".cls", ".pdf"}
MAX_CANDIDATE_BYTES = 25 * 1024 * 1024
LOCAL_TABLES_TO_VALIDATE = (
    "outputs/tables/article_canonical_representation_table.csv",
)
LOCAL_PATH_PATTERNS = (
    re.compile(r"[A-Za-z]:[\\/]Users[\\/]", re.IGNORECASE),
    re.compile(r"/(?:Users|home)/[^/]+/", re.IGNORECASE),
)
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?:api[_-]?key|secret|password|passwd)\s*[:=]\s*[^\s]+", re.IGNORECASE),
    re.compile(r"(?:ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,})"),
)


def git_lines(*args: str) -> list[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def normalise_git_path(value: str) -> str:
    return value.replace("\\", "/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-license",
        action="store_true",
        help="Fail if no root LICENSE file has been selected.",
    )
    return parser.parse_args()


def validate_csv_width(path: Path, errors: list[str]) -> None:
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
    except (UnicodeDecodeError, OSError, csv.Error) as exc:
        errors.append(f"could not validate CSV {path.relative_to(ROOT)}: {exc}")
        return
    if not rows:
        errors.append(f"empty CSV file: {path.relative_to(ROOT)}")
        return
    expected = len(rows[0])
    for line_number, row in enumerate(rows[1:], start=2):
        if len(row) != expected:
            errors.append(
                f"inconsistent CSV width in {path.relative_to(ROOT)} line {line_number}: "
                f"expected {expected}, found {len(row)}"
            )


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    warnings: list[str] = []
    tracked = [normalise_git_path(line) for line in git_lines("ls-files")]
    status_lines = git_lines("status", "--porcelain=v1", "--untracked-files=all")
    candidates = set(tracked)
    for line in status_lines:
        if len(line) < 4:
            continue
        if "D" in line[:2]:
            continue
        status_path = line[3:]
        if " -> " in status_path:
            status_path = status_path.split(" -> ", 1)[1]
        candidates.add(normalise_git_path(status_path.strip('"')))

    for path in REQUIRED_FILES:
        if not (ROOT / path).exists():
            errors.append(f"missing required file: {path}")

    licence_files = [ROOT / name for name in ("LICENSE", "LICENSE.txt", "LICENSE.md")]
    if not any(path.exists() for path in licence_files):
        message = "software licence has not been selected"
        if args.require_license:
            errors.append(message)
        else:
            warnings.append(message)

    for path in candidates:
        if path.startswith(FORBIDDEN_PREFIXES):
            errors.append(f"forbidden tracked path: {path}")

        file_path = ROOT / path
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() in FORBIDDEN_DATA_SUFFIXES:
            errors.append(f"forbidden data or model file: {path}")

        if file_path.suffix.lower() in FORBIDDEN_PUBLIC_SUFFIXES:
            errors.append(f"forbidden manuscript or PDF artefact: {path}")

        if file_path.stat().st_size > MAX_CANDIDATE_BYTES:
            errors.append(f"candidate file exceeds 25 MiB: {path}")

        if file_path.suffix.lower() == ".csv":
            validate_csv_width(file_path, errors)

        try:
            content = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        for pattern in LOCAL_PATH_PATTERNS:
            if pattern.search(content):
                errors.append(f"local absolute path found in {path}")
                break

        for pattern in SECRET_PATTERNS:
            if pattern.search(content):
                errors.append(f"possible secret pattern found in {path}")
                break

    for path in LOCAL_TABLES_TO_VALIDATE:
        file_path = ROOT / path
        if file_path.exists():
            validate_csv_width(file_path, errors)

    if warnings:
        print("Public-release audit warnings:")
        for warning in warnings:
            print(f"- {warning}")

    if errors:
        print("Public-release audit failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Public-release audit passed for {len(candidates)} candidate files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
