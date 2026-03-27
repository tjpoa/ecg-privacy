from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import pandas as pd

from config import SEGMENT_FEATURES_DATASET_DIR, WFDB_RECORDS_DIR
from data_loading import load_signal_fast
from feature_extraction import extract_segment_features, map_clinical_label
from segmentation import read_metadata
from preprocessing import preprocess_record


def build_segment_feature_rows(
    record_path: Path,
    utility_codes: set[str],
    positive_label: str,
    negative_label: str,
    lowcut: float,
    highcut: float,
    order: int,
    window_sec: float,
    step_sec: float,
):
    metadata = read_metadata(record_path)
    processed = preprocess_record(
        record_path=record_path.with_suffix(""),
        loader=load_signal_fast,
        lowcut=lowcut,
        highcut=highcut,
        order=order,
        window_sec=window_sec,
        step_sec=step_sec,
    )

    rows = []
    for idx, ((start, end), segment) in enumerate(zip(processed["segment_ranges"], processed["segments"])):
        row = {
            "patient_id": metadata["patient_id"],
            "segment_id": f"{metadata['patient_id']}_seg_{idx:04d}",
            "label": metadata["label"],
            "utility_label": map_clinical_label(
                metadata["label"],
                positive_codes=utility_codes,
                positive_label=positive_label,
                negative_label=negative_label,
            ),
            "segment_ref": idx,
            "start_sample": start,
            "end_sample": end,
        }
        row.update(
            extract_segment_features(
                segment,
                leads=processed["leads"],
                fs=processed["fs"],
                include_rr_features=False,
            )
        )
        rows.append(row)

    return rows


def iter_batches(items: list[Path], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start // batch_size, items[start : start + batch_size]


def main():
    parser = argparse.ArgumentParser(
        description="Build a segment-level ECG features dataset in batches to avoid notebook memory issues."
    )
    parser.add_argument("--record-limit", type=int, default=None, help="Optional limit on the number of ECG records.")
    parser.add_argument("--batch-size", type=int, default=250, help="Number of records to process per output chunk.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SEGMENT_FEATURES_DATASET_DIR,
        help="Directory where chunk files and metadata will be written.",
    )
    parser.add_argument("--lowcut", type=float, default=0.5)
    parser.add_argument("--highcut", type=float, default=40.0)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--step-sec", type=float, default=0.5)
    parser.add_argument(
        "--atrial-codes",
        nargs="+",
        default=["164889003", "164890007"],
        help="SNOMED codes treated as positive for the utility task.",
    )
    args = parser.parse_args()

    records = sorted(WFDB_RECORDS_DIR.rglob("*.hea"))
    if args.record_limit is not None:
        records = records[: args.record_limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at_epoch": time.time(),
        "record_count": len(records),
        "batch_size": args.batch_size,
        "window_sec": args.window_sec,
        "step_sec": args.step_sec,
        "lowcut": args.lowcut,
        "highcut": args.highcut,
        "order": args.order,
        "atrial_codes": list(args.atrial_codes),
        "chunks": [],
        "errors_file": "errors.csv",
    }

    error_rows = []
    utility_codes = set(args.atrial_codes)
    total_batches = max(1, math.ceil(len(records) / args.batch_size))
    start_time = time.time()

    for batch_idx, batch_records in iter_batches(records, args.batch_size):
        batch_rows = []
        batch_start = time.time()

        for record_path in batch_records:
            try:
                batch_rows.extend(
                    build_segment_feature_rows(
                        record_path=record_path,
                        utility_codes=utility_codes,
                        positive_label="Atrial",
                        negative_label="Non-Atrial",
                        lowcut=args.lowcut,
                        highcut=args.highcut,
                        order=args.order,
                        window_sec=args.window_sec,
                        step_sec=args.step_sec,
                    )
                )
            except Exception as exc:  # pragma: no cover - defensive batch runner
                error_rows.append(
                    {
                        "record_path": str(record_path),
                        "patient_id": record_path.stem,
                        "error": str(exc),
                    }
                )

        chunk_df = pd.DataFrame(batch_rows)
        chunk_name = f"segment_features_chunk_{batch_idx:04d}.csv.gz"
        chunk_path = output_dir / chunk_name
        chunk_df.to_csv(chunk_path, index=False, compression="gzip")

        manifest["chunks"].append(
            {
                "chunk_file": chunk_name,
                "records_in_batch": len(batch_records),
                "segments_in_chunk": int(len(chunk_df)),
            }
        )

        elapsed = time.time() - batch_start
        print(
            f"Batch {batch_idx + 1}/{total_batches} done: "
            f"{len(batch_records)} records, {len(chunk_df)} segments, {elapsed:.1f}s"
        )

    errors_df = pd.DataFrame(error_rows)
    errors_df.to_csv(output_dir / "errors.csv", index=False)

    manifest["total_segments"] = int(sum(chunk["segments_in_chunk"] for chunk in manifest["chunks"]))
    manifest["error_count"] = int(len(errors_df))
    manifest["elapsed_seconds"] = round(time.time() - start_time, 2)

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"Completed dataset build in {manifest['elapsed_seconds']:.1f}s. "
        f"Chunks: {len(manifest['chunks'])}, segments: {manifest['total_segments']}, errors: {manifest['error_count']}"
    )


if __name__ == "__main__":
    main()
