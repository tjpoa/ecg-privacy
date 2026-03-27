from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from config import OUTPUTS_TABLES_DIR, WINDOW_OVERLAP_SWEEP_DIR
from modeling import run_linkability_baselines, run_utility_baselines


def parse_config(spec: str) -> tuple[float, float]:
    try:
        window_text, overlap_text = spec.split(",", maxsplit=1)
        window_sec = float(window_text)
        overlap = float(overlap_text)
    except ValueError as exc:  # pragma: no cover - CLI guard
        raise ValueError(f"Invalid config '{spec}'. Use the format 'window,overlap', for example '2.0,0.75'.") from exc

    if window_sec <= 0:
        raise ValueError(f"window must be positive in config '{spec}'.")
    if not (0 <= overlap < 1):
        raise ValueError(f"overlap must be in [0, 1) in config '{spec}'.")

    return window_sec, overlap


def format_value(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def build_output_dir(output_root: Path, window_sec: float, overlap: float) -> Path:
    return output_root / f"w{format_value(window_sec)}_o{format_value(overlap)}"


def run_builder(
    python_executable: str,
    record_limit: int | None,
    batch_size: int,
    output_dir: Path,
    window_sec: float,
    overlap: float,
    lowcut: float,
    highcut: float,
    order: int,
    atrial_codes: list[str],
) -> None:
    step_sec = window_sec * (1 - overlap)
    command = [
        python_executable,
        "src/build_segment_features_dataset.py",
        "--batch-size",
        str(batch_size),
        "--output-dir",
        str(output_dir),
        "--window-sec",
        str(window_sec),
        "--step-sec",
        str(step_sec),
        "--lowcut",
        str(lowcut),
        "--highcut",
        str(highcut),
        "--order",
        str(order),
        "--atrial-codes",
        *atrial_codes,
    ]
    if record_limit is not None:
        command.extend(["--record-limit", str(record_limit)])

    subprocess.run(command, check=True)


def load_feature_dataset(dataset_dir: Path) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    manifest_path = dataset_dir / "manifest.json"
    errors_path = dataset_dir / "errors.csv"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunk_files = [dataset_dir / chunk["chunk_file"] for chunk in manifest["chunks"]]
    features_df = pd.concat(
        [pd.read_csv(chunk_file, compression="gzip", low_memory=False) for chunk_file in chunk_files],
        ignore_index=True,
    )
    if errors_path.exists() and errors_path.stat().st_size > 0:
        try:
            errors_df = pd.read_csv(errors_path)
        except EmptyDataError:
            errors_df = pd.DataFrame()
    else:
        errors_df = pd.DataFrame()
    return features_df, manifest, errors_df


def summarize_results(
    utility_summary: pd.DataFrame,
    linkability_summary: pd.DataFrame,
    config_name: str,
    manifest: dict,
    errors_df: pd.DataFrame,
) -> pd.DataFrame:
    utility_rows = utility_summary.copy()
    utility_rows["task"] = "utility"
    linkability_rows = linkability_summary.copy()
    linkability_rows["task"] = "linkability"

    summary = pd.concat([utility_rows, linkability_rows], ignore_index=True)
    summary.insert(0, "config", config_name)
    summary.insert(1, "window_sec", manifest.get("window_sec"))
    summary.insert(2, "step_sec", manifest.get("step_sec"))
    summary.insert(3, "record_count", manifest.get("record_count"))
    summary.insert(4, "segment_count", manifest.get("total_segments"))
    summary.insert(5, "error_count", len(errors_df))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a small sweep of window/overlap configurations and compare baseline metrics."
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["1.0,0.5", "2.0,0.5", "2.0,0.75", "3.0,0.5"],
        help="List of configs in the format 'window,overlap'. Example: 2.0,0.75",
    )
    parser.add_argument("--record-limit", type=int, default=1000, help="Number of records used for the sweep.")
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--output-root", type=Path, default=WINDOW_OVERLAP_SWEEP_DIR)
    parser.add_argument("--lowcut", type=float, default=0.5)
    parser.add_argument("--highcut", type=float, default=40.0)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-pairs", type=int, default=2000)
    parser.add_argument("--min-segment-gap", type=int, default=4)
    parser.add_argument(
        "--utility-models",
        nargs="+",
        default=["LogisticRegression", "XGBoost"],
    )
    parser.add_argument(
        "--linkability-models",
        nargs="+",
        default=["LogisticRegression", "XGBoost"],
    )
    parser.add_argument(
        "--atrial-codes",
        nargs="+",
        default=["164889003", "164890007"],
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse an existing dataset folder when manifest.json is already present.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    OUTPUTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    summary_frames = []
    python_executable = sys.executable
    started = time.time()

    for spec in args.configs:
        window_sec, overlap = parse_config(spec)
        config_name = f"w{format_value(window_sec)}_o{format_value(overlap)}"
        dataset_dir = build_output_dir(output_root, window_sec, overlap)

        print()
        print(f"=== Running config {config_name} ===")

        if args.skip_existing and (dataset_dir / "manifest.json").exists():
            print(f"Reusing existing dataset at {dataset_dir}")
        else:
            run_builder(
                python_executable=python_executable,
                record_limit=args.record_limit,
                batch_size=args.batch_size,
                output_dir=dataset_dir,
                window_sec=window_sec,
                overlap=overlap,
                lowcut=args.lowcut,
                highcut=args.highcut,
                order=args.order,
                atrial_codes=args.atrial_codes,
            )

        features_df, manifest, errors_df = load_feature_dataset(dataset_dir)
        print(f"Loaded {features_df.shape[0]} segments with {features_df.shape[1]} columns from {dataset_dir.name}")

        utility_results = run_utility_baselines(
            features_df=features_df,
            target_col="utility_label",
            group_col="patient_id",
            test_size=args.test_size,
            random_state=args.random_state,
            models=args.utility_models,
        )
        linkability_results = run_linkability_baselines(
            features_df=features_df,
            group_col="patient_id",
            test_size=args.test_size,
            random_state=args.random_state,
            max_pairs=args.max_pairs,
            representation="absdiff",
            min_segment_gap=args.min_segment_gap,
        )

        summary_df = summarize_results(
            utility_summary=utility_results["summary_df"],
            linkability_summary=linkability_results["summary_df"],
            config_name=config_name,
            manifest=manifest,
            errors_df=errors_df,
        )
        summary_frames.append(summary_df)

        print(summary_df[["config", "task", "model", "f1_score", "balanced_accuracy", "roc_auc", "pr_auc"]])

    all_results_df = pd.concat(summary_frames, ignore_index=True)
    output_csv = OUTPUTS_TABLES_DIR / "baseline_sweep_summary.csv"
    all_results_df.to_csv(output_csv, index=False)

    print()
    print(f"Sweep completed in {time.time() - started:.1f}s")
    print(f"Datasets written under {output_root}")
    print(f"Summary written to {output_csv}")


if __name__ == "__main__":
    main()
