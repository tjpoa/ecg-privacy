from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from config import OUTPUTS_TABLES_DIR
    from run_encoder_dp_operational_linkage_eval import summarize
except ImportError:  # pragma: no cover
    from .config import OUTPUTS_TABLES_DIR
    from .run_encoder_dp_operational_linkage_eval import summarize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolidate checkpointed encoder+DP operational seeds.")
    parser.add_argument("--run-names", nargs="+", required=True)
    parser.add_argument("--output-run-name", default="encoder_dp_operational_dim8_multiseed")
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_TABLES_DIR)
    return parser.parse_args()


def load_runs(args: argparse.Namespace, suffix: str) -> pd.DataFrame:
    paths = [args.output_dir / f"{name}_{suffix}.csv" for name in args.run_names]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {suffix} checkpoints: {missing}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def main() -> None:
    args = parse_args()
    metrics = load_runs(args, "metrics")
    splits = load_runs(args, "splits")
    expected_seeds = [42, 123, 456]
    expected_transforms = {
        "encoder_identity",
        "encoder_dp_gaussian_eps100_clip2",
        "encoder_dp_gaussian_eps50_clip2",
        "encoder_dp_gaussian_eps50_clip4",
        "encoder_dp_gaussian_eps20_clip4",
    }
    expected_galleries = {99, 999}

    if sorted(int(value) for value in metrics["seed"].unique()) != expected_seeds:
        raise AssertionError("Operational checkpoints do not contain the expected seeds.")
    if set(metrics["transform_name"]) != expected_transforms:
        raise AssertionError("Operational checkpoints do not contain the expected transforms.")
    if set(metrics["gallery_negatives"].astype(int)) != expected_galleries:
        raise AssertionError("Operational checkpoints do not contain both gallery sizes.")
    counts = metrics.groupby(["transform_name", "gallery_negatives"]).size()
    if not counts.eq(len(expected_seeds)).all():
        raise AssertionError(f"Incomplete operational conditions: {counts.to_dict()}")
    if len(metrics) != len(expected_seeds) * len(expected_transforms) * len(expected_galleries):
        raise AssertionError(f"Expected 30 operational rows, observed {len(metrics)}")
    if not splits["group_overlap"].eq(0).all():
        raise AssertionError("At least one operational split reports retained-record overlap.")

    summary = summarize(metrics)
    prefix = args.output_dir / args.output_run_name
    metrics.to_csv(Path(f"{prefix}_metrics.csv"), index=False)
    summary.to_csv(Path(f"{prefix}_summary.csv"), index=False)
    splits.to_csv(Path(f"{prefix}_splits.csv"), index=False)
    Path(f"{prefix}_protocol.json").write_text(json.dumps({
        "source_runs": args.run_names,
        "seeds": expected_seeds,
        "transforms": sorted(expected_transforms),
        "gallery_negatives": sorted(expected_galleries),
        "validation": "All 30 expected rows are present and every retained-record split has zero overlap.",
    }, indent=2), encoding="utf-8")
    print("Metrics rows:", len(metrics))
    print("Summary rows:", len(summary))
    print("Saved prefix:", prefix)


if __name__ == "__main__":
    main()
