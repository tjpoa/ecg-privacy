from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from config import OUTPUTS_TABLES_DIR
    from run_representation_dimension_comparison import summarize_metrics
except ImportError:  # pragma: no cover
    from .config import OUTPUTS_TABLES_DIR
    from .run_representation_dimension_comparison import summarize_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolidate checkpointed representation-dimension seeds.")
    parser.add_argument("--run-names", nargs="+", required=True)
    parser.add_argument("--output-run-name", default="representation_dimension_final_multiseed")
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = load_runs(args, "metrics")
    predictions = load_runs(args, "utility_predictions")
    splits = load_runs(args, "splits")

    expected_seeds = sorted({42, 123, 456})
    observed_seeds = sorted(int(value) for value in metrics["seed"].unique())
    if observed_seeds != expected_seeds:
        raise AssertionError(f"Expected seeds {expected_seeds}, observed {observed_seeds}")
    expected_configs = {
        ("identity", 208),
        ("supervised_encoder", 8),
        ("supervised_encoder", 64),
        ("pca", 8),
        ("pca", 64),
        ("random_projection", 8),
        ("random_projection", 64),
    }
    observed_configs = set(zip(metrics["method"], metrics["dimension"].astype(int)))
    if observed_configs != expected_configs:
        raise AssertionError(f"Unexpected representation configurations: {observed_configs}")
    counts = metrics.groupby(["method", "dimension"]).size()
    if not counts.eq(len(expected_seeds)).all():
        raise AssertionError(f"Incomplete multiseed configurations: {counts.to_dict()}")
    if not splits["group_overlap"].eq(0).all():
        raise AssertionError("At least one ECG split reports retained-record overlap.")

    summary = summarize_metrics(metrics)
    prefix = args.output_dir / args.output_run_name
    metrics.to_csv(Path(f"{prefix}_metrics.csv"), index=False)
    summary.to_csv(Path(f"{prefix}_summary.csv"), index=False)
    predictions.to_csv(Path(f"{prefix}_utility_predictions.csv"), index=False)
    splits.to_csv(Path(f"{prefix}_splits.csv"), index=False)
    Path(f"{prefix}_protocol.json").write_text(json.dumps({
        "source_runs": args.run_names,
        "seeds": expected_seeds,
        "configurations": sorted([list(value) for value in expected_configs]),
        "validation": "Every representation has one evaluation per seed and every split has zero group overlap.",
    }, indent=2), encoding="utf-8")
    print("Consolidated seeds:", expected_seeds)
    print("Configurations:", len(expected_configs))
    print("Metrics rows:", len(metrics))
    print("Saved prefix:", prefix)


if __name__ == "__main__":
    main()
