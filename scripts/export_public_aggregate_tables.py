"""Refresh the curated aggregate tables included in the public release."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "tables"
DESTINATION = ROOT / "reproducibility" / "tables"
FILES = (
    "main_utility_linkability_tradeoff_figure_data.csv",
    "main_paired_uncertainty_summary.csv",
    "article_canonical_representation_table.csv",
    "absolute_attribute_risk_summary.csv",
    "privacy_attack_extensions_final_protocol_attribute_bootstrap_effects.csv",
    "privacy_attack_extensions_final_protocol_reconstruction_article_table.csv",
    "conditioned_reconstruction_final_protocol_multiseed_article_table.csv",
    "article_configuration_selection_evidence.csv",
    "article_model_configuration_audit.csv",
    "computational_environment_versions.csv",
)
FORBIDDEN_COLUMNS = {
    "patient_id",
    "record_id",
    "retained_record_id",
    "segment_id",
    "waveform",
    "prediction",
}


def validate_source(path: Path) -> None:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"Empty aggregate table: {path}")
    width = len(rows[0])
    for line_number, row in enumerate(rows[1:], start=2):
        if len(row) != width:
            raise ValueError(
                f"Inconsistent CSV width in {path} line {line_number}: "
                f"expected {width}, found {len(row)}"
            )
    columns = {column.strip().lower() for column in rows[0]}
    forbidden = sorted(columns & FORBIDDEN_COLUMNS)
    if forbidden:
        raise ValueError(f"Record-level columns are not public: {path}: {forbidden}")


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        source = SOURCE / name
        if not source.exists():
            raise FileNotFoundError(source)
        validate_source(source)
        destination = DESTINATION / name
        shutil.copyfile(source, destination)
        validate_source(destination)
        print(f"Updated {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
