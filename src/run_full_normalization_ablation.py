from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from config import FINAL_SEGMENT_FEATURES_DIR, OUTPUTS_TABLES_DIR, WFDB_RECORDS_DIR
    from data_loading import load_signal_fast
    from feature_extraction import extract_segment_features, map_clinical_label
    from modeling import (
        evaluate_utility_model_on_split,
        get_feature_columns,
        run_linkability_baselines_on_split,
        split_segments_by_patient_stratified,
    )
    from preprocessing import bandpass_filter, segment_signal, validate_signal, zscore_normalize
    from segmentation import read_metadata
except ImportError:  # pragma: no cover - package import fallback
    from .config import FINAL_SEGMENT_FEATURES_DIR, OUTPUTS_TABLES_DIR, WFDB_RECORDS_DIR
    from .data_loading import load_signal_fast
    from .feature_extraction import extract_segment_features, map_clinical_label
    from .modeling import (
        evaluate_utility_model_on_split,
        get_feature_columns,
        run_linkability_baselines_on_split,
        split_segments_by_patient_stratified,
    )
    from .preprocessing import bandpass_filter, segment_signal, validate_signal, zscore_normalize
    from .segmentation import read_metadata


ATRIAL_CODES = {"164889003", "164890007"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Full-dataset ablation comparing complete-record and per-segment ECG standardisation."
        )
    )
    parser.add_argument("--records-dir", type=Path, default=WFDB_RECORDS_DIR)
    parser.add_argument("--record-feature-dir", type=Path, default=FINAL_SEGMENT_FEATURES_DIR)
    parser.add_argument(
        "--segment-feature-dir",
        type=Path,
        default=FINAL_SEGMENT_FEATURES_DIR.parent / "segment_normalized_features_dataset",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_TABLES_DIR / "normalization_sensitivity_full")
    parser.add_argument("--record-limit", type=int, default=0, help="Use 0 for every available record.")
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--executor", choices=["process", "thread"], default="process")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--force-rebuild", action="store_true")
    return parser.parse_args()


def extract_segment_normalized_record(record_path: Path) -> tuple[list[dict[str, object]], dict | None]:
    try:
        signal, fs, leads = load_signal_fast(record_path)
        checks = validate_signal(signal, fs, leads)
        if not checks["is_valid"]:
            raise ValueError(f"invalid record: {checks}")
        filtered = bandpass_filter(signal, fs=fs, lowcut=0.5, highcut=40.0, order=4)
        segments, ranges = segment_signal(filtered, fs=fs, window_sec=2.0, step_sec=1.0)
        segments = np.stack([zscore_normalize(segment)[0] for segment in segments], axis=0)
        metadata = read_metadata(record_path)
        utility_label = map_clinical_label(
            metadata["label"],
            positive_codes=ATRIAL_CODES,
            positive_label="Atrial",
            negative_label="Non-Atrial",
        )
        rows: list[dict[str, object]] = []
        for segment_ref, ((start, end), segment) in enumerate(zip(ranges, segments)):
            row = {
                "patient_id": metadata["patient_id"],
                "segment_id": f"{metadata['patient_id']}_seg_{segment_ref:04d}",
                "label": metadata["label"],
                "utility_label": utility_label,
                "segment_ref": segment_ref,
                "start_sample": start,
                "end_sample": end,
            }
            row.update(extract_segment_features(segment, leads=leads, fs=fs, include_rr_features=False))
            rows.append(row)
        return rows, None
    except Exception as exc:  # pragma: no cover - defensive full-dataset runner
        return [], {"record_path": str(record_path), "patient_id": record_path.stem, "error": str(exc)}


def build_segment_normalized_dataset(args: argparse.Namespace, records: list[Path]) -> dict:
    manifest_path = args.segment_feature_dir / "manifest.json"
    if manifest_path.exists() and not args.force_rebuild:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("record_count", -1)) == len(records):
            print("Reusing segment-normalized feature cache:", args.segment_feature_dir, flush=True)
            return manifest

    args.segment_feature_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    manifest = {
        "created_at_epoch": started,
        "record_count": len(records),
        "batch_size": args.batch_size,
        "normalization": "per-segment, per-lead z-score after filtering",
        "window_sec": 2.0,
        "step_sec": 1.0,
        "lowcut": 0.5,
        "highcut": 40.0,
        "order": 4,
        "atrial_codes": sorted(ATRIAL_CODES),
        "chunks": [],
        "errors_file": "errors.csv",
    }
    errors: list[dict] = []
    total_batches = math.ceil(len(records) / args.batch_size)
    executor_class = ProcessPoolExecutor if args.executor == "process" else ThreadPoolExecutor
    with executor_class(max_workers=args.workers) as executor:
        for batch_index in range(total_batches):
            batch_started = time.time()
            start = batch_index * args.batch_size
            batch_records = records[start : start + args.batch_size]
            results = list(executor.map(extract_segment_normalized_record, batch_records))
            rows: list[dict[str, object]] = []
            for record_rows, error in results:
                rows.extend(record_rows)
                if error is not None:
                    errors.append(error)
            chunk_name = f"segment_features_chunk_{batch_index:04d}.csv.gz"
            pd.DataFrame(rows).to_csv(
                args.segment_feature_dir / chunk_name,
                index=False,
                compression="gzip",
            )
            manifest["chunks"].append(
                {
                    "chunk_file": chunk_name,
                    "records_in_batch": len(batch_records),
                    "segments_in_chunk": len(rows),
                }
            )
            print(
                f"Batch {batch_index + 1}/{total_batches}: {len(batch_records)} records, "
                f"{len(rows)} segments, {time.time() - batch_started:.1f}s",
                flush=True,
            )
    pd.DataFrame(errors).to_csv(args.segment_feature_dir / "errors.csv", index=False)
    manifest["total_segments"] = int(sum(item["segments_in_chunk"] for item in manifest["chunks"]))
    manifest["error_count"] = len(errors)
    manifest["elapsed_seconds"] = round(time.time() - started, 2)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def chunk_paths(dataset_dir: Path) -> list[Path]:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    return [dataset_dir / item["chunk_file"] for item in manifest["chunks"]]


def load_identifiers(dataset_dir: Path) -> set[str]:
    values: set[str] = set()
    for path in chunk_paths(dataset_dir):
        values.update(pd.read_csv(path, compression="gzip", usecols=["patient_id"])["patient_id"].astype(str))
    return values


def load_features(dataset_dir: Path, common_ids: set[str]) -> pd.DataFrame:
    frame = pd.concat(
        [pd.read_csv(path, compression="gzip", low_memory=False) for path in chunk_paths(dataset_dir)],
        ignore_index=True,
    )
    frame["patient_id"] = frame["patient_id"].astype(str)
    return frame[frame["patient_id"].isin(common_ids)].reset_index(drop=True)


def evaluate_condition(features: pd.DataFrame, normalization: str, seed: int) -> dict[str, object]:
    train_df, test_df = split_segments_by_patient_stratified(
        features,
        group_col="patient_id",
        label_col="utility_label",
        test_size=0.2,
        random_state=seed,
    )
    feature_columns = get_feature_columns(features)
    utility = evaluate_utility_model_on_split(
        train_df=train_df,
        test_df=test_df,
        model_name="LogisticRegression",
        target_col="utility_label",
        feature_columns=feature_columns,
    )["evaluation"]
    linkage = run_linkability_baselines_on_split(
        train_df=train_df,
        test_df=test_df,
        max_pairs=2000,
        representation="absdiff",
        random_state=seed,
        min_segment_gap=4,
        max_positive_pairs_per_patient=3,
        negative_strategy="random",
        hard_negative_pool_size=5,
        include_distance_baselines=False,
    )
    xgb = linkage["summary_df"].loc[lambda frame: frame["model"] == "XGBoost"].iloc[0]
    return {
        "normalization": normalization,
        "seed": seed,
        "records": int(features["patient_id"].nunique()),
        "segments": int(len(features)),
        "train_records": int(train_df["patient_id"].nunique()),
        "test_records": int(test_df["patient_id"].nunique()),
        "utility_balanced_accuracy": float(utility["balanced_accuracy"]),
        "utility_f1": float(utility["f1_score"]),
        "linkability_roc_auc": float(xgb["roc_auc"]),
        "linkability_balanced_accuracy": float(xgb["balanced_accuracy"]),
        "link_train_pairs": int(len(linkage["train_pair_df"])),
        "link_test_pairs": int(len(linkage["test_pair_df"])),
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    records = sorted(args.records_dir.rglob("*.hea"))
    if args.record_limit > 0:
        records = records[: args.record_limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    segment_manifest = build_segment_normalized_dataset(args, records)

    common_ids = load_identifiers(args.record_feature_dir).intersection(load_identifiers(args.segment_feature_dir))
    print(f"Common retained records: {len(common_ids)}", flush=True)
    rows: list[dict[str, object]] = []
    for normalization, dataset_dir in [
        ("complete_record", args.record_feature_dir),
        ("per_segment", args.segment_feature_dir),
    ]:
        print(f"Loading and evaluating {normalization} features...", flush=True)
        features = load_features(dataset_dir, common_ids)
        for seed in args.seeds:
            metrics = evaluate_condition(features, normalization=normalization, seed=seed)
            rows.append(metrics)
            print(metrics, flush=True)
        del features

    metrics = pd.DataFrame(rows)
    summary = metrics.groupby("normalization", as_index=False)[
        ["utility_balanced_accuracy", "utility_f1", "linkability_roc_auc", "linkability_balanced_accuracy"]
    ].agg(["mean", "std"])
    summary.columns = [
        "_".join(part for part in column if part) if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    metrics_path = args.output_dir / "normalization_sensitivity_full_metrics.csv"
    summary_path = args.output_dir / "normalization_sensitivity_full_summary.csv"
    protocol_path = args.output_dir / "normalization_sensitivity_full_protocol.json"
    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    protocol_path.write_text(
        json.dumps(
            {
                "records_requested": len(records),
                "common_retained_records": len(common_ids),
                "seeds": args.seeds,
                "record_feature_dir": str(args.record_feature_dir),
                "segment_feature_dir": str(args.segment_feature_dir),
                "executor": args.executor,
                "workers": args.workers,
                "segment_feature_manifest": segment_manifest,
                "split": "Same deterministic retained-record split for each normalization and seed.",
                "elapsed_seconds": round(time.time() - started, 2),
                "outputs": {"metrics": str(metrics_path), "summary": str(summary_path)},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
