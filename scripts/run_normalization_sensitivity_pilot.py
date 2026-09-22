from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loading import load_signal_fast  # noqa: E402
from feature_extraction import extract_segment_features, map_clinical_label  # noqa: E402
from modeling import (  # noqa: E402
    evaluate_utility_model_on_split,
    get_feature_columns,
    run_linkability_baselines_on_split,
    split_segments_by_patient_stratified,
)
from preprocessing import (  # noqa: E402
    bandpass_filter,
    segment_signal,
    validate_signal,
    zscore_normalize,
)
from segmentation import read_metadata  # noqa: E402


ATRIAL_CODES = {"164889003", "164890007"}


def build_feature_frame(records: list[Path], normalization: str) -> tuple[pd.DataFrame, int]:
    rows: list[dict[str, object]] = []
    errors = 0

    for record_path in records:
        try:
            signal, fs, leads = load_signal_fast(record_path)
            checks = validate_signal(signal, fs, leads)
            if not checks["is_valid"]:
                raise ValueError(f"invalid record: {checks}")

            filtered = bandpass_filter(signal, fs=fs, lowcut=0.5, highcut=40.0, order=4)
            if normalization == "record":
                normalized, _, _ = zscore_normalize(filtered)
                segments, ranges = segment_signal(normalized, fs=fs, window_sec=2.0, step_sec=1.0)
            elif normalization == "segment":
                segments, ranges = segment_signal(filtered, fs=fs, window_sec=2.0, step_sec=1.0)
                segments = np.stack([zscore_normalize(segment)[0] for segment in segments], axis=0)
            else:
                raise ValueError(f"unsupported normalization: {normalization}")

            metadata = read_metadata(record_path)
            utility_label = map_clinical_label(
                metadata["label"],
                positive_codes=ATRIAL_CODES,
                positive_label="Atrial",
                negative_label="Non-Atrial",
            )
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
        except Exception:
            errors += 1

    return pd.DataFrame(rows), errors


def evaluate_frame(features_df: pd.DataFrame, seed: int) -> dict[str, object]:
    train_df, test_df = split_segments_by_patient_stratified(
        features_df,
        group_col="patient_id",
        label_col="utility_label",
        test_size=0.2,
        random_state=seed,
    )
    feature_columns = get_feature_columns(features_df)
    utility = evaluate_utility_model_on_split(
        train_df=train_df,
        test_df=test_df,
        model_name="LogisticRegression",
        target_col="utility_label",
        feature_columns=feature_columns,
    )["evaluation"]
    linkability = run_linkability_baselines_on_split(
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
    xgb = linkability["summary_df"].loc[lambda frame: frame["model"] == "XGBoost"].iloc[0]
    return {
        "records": int(features_df["patient_id"].nunique()),
        "segments": int(len(features_df)),
        "train_records": int(train_df["patient_id"].nunique()),
        "test_records": int(test_df["patient_id"].nunique()),
        "utility_balanced_accuracy": float(utility["balanced_accuracy"]),
        "utility_f1": float(utility["f1_score"]),
        "linkability_roc_auc": float(xgb["roc_auc"]),
        "linkability_balanced_accuracy": float(xgb["balanced_accuracy"]),
        "link_train_pairs": int(len(linkability["train_pair_df"])),
        "link_test_pairs": int(len(linkability["test_pair_df"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-limit", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "tables" / "normalization_sensitivity_pilot",
    )
    args = parser.parse_args()

    raw_dir = PROJECT_ROOT / "data" / "raw" / "WFDBRecords"
    records = sorted(raw_dir.rglob("*.hea"))[: args.record_limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    result_rows: list[dict[str, object]] = []
    for normalization in ("record", "segment"):
        print(f"Building {normalization}-normalised features for {len(records)} records...", flush=True)
        features_df, errors = build_feature_frame(records, normalization=normalization)
        for seed in args.seeds:
            metrics = evaluate_frame(features_df, seed=seed)
            metrics.update({"normalization": normalization, "errors": errors, "seed": seed})
            result_rows.append(metrics)
            print(metrics, flush=True)

    results_df = pd.DataFrame(result_rows)
    results_df.to_csv(args.output_dir / "normalization_sensitivity_pilot.csv", index=False)
    protocol = {
        "record_limit": args.record_limit,
        "records_requested": len(records),
        "seeds": args.seeds,
        "window_sec": 2.0,
        "step_sec": 1.0,
        "filter": {"lowcut_hz": 0.5, "highcut_hz": 40.0, "order": 4},
        "atrial_codes": sorted(ATRIAL_CODES),
        "test_size": 0.2,
        "link_max_pairs": 2000,
        "link_min_segment_gap": 4,
        "link_max_positive_pairs_per_record": 3,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (args.output_dir / "normalization_sensitivity_pilot_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(results_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
