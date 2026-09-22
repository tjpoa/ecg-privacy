from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "manuscript" / "overleaf_jbhi_submission" / "main.tex"
MIRROR = ROOT / "manuscript" / "jbhi_ecg_article.tex"

PATH_REWRITES = {
    r"\graphicspath{{figures/}}": r"\graphicspath{{overleaf_jbhi_submission/figures/}}",
    r"\bibliography{references}": r"\bibliography{overleaf_jbhi_submission/references}",
}


def expected_mirror() -> str:
    text = CANONICAL.read_text(encoding="utf-8")
    for canonical_value, mirror_value in PATH_REWRITES.items():
        count = text.count(canonical_value)
        if count != 1:
            raise ValueError(
                f"Expected exactly one occurrence of {canonical_value!r} in {CANONICAL}; found {count}."
            )
        text = text.replace(canonical_value, mirror_value)
    return text


def write_mirror(text: str) -> None:
    with MIRROR.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synchronize the repository-level ECG manuscript mirror from the canonical Overleaf source."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Return a non-zero status when the mirror is not synchronized.",
    )
    args = parser.parse_args()

    expected = expected_mirror()
    current = MIRROR.read_text(encoding="utf-8") if MIRROR.exists() else None
    if current == expected:
        print(f"Synchronized: {MIRROR}")
        return
    if args.check:
        raise SystemExit(f"Out of sync: {MIRROR}")

    write_mirror(expected)
    if MIRROR.read_text(encoding="utf-8") != expected:
        raise RuntimeError(f"Synchronization verification failed: {MIRROR}")
    print(f"Updated: {MIRROR}")


if __name__ == "__main__":
    main()
