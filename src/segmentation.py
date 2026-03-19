from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing import preprocess_record


def read_metadata(hea_path):
    hea_path = Path(hea_path).with_suffix(".hea")
    metadata = {"patient_id": hea_path.stem, "label": None}

    with open(hea_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#Dx:"):
                metadata["label"] = line.split(":", 1)[1].strip()

    return metadata


def build_segment_metadata(patient_id, label, segment_ranges):
    rows = []

    for idx, (start, end) in enumerate(segment_ranges):
        rows.append({
            "patient_id": patient_id,
            "segment_id": f"{patient_id}_seg_{idx:04d}",
            "label": label,
            "segment_ref": idx,
            "start_sample": start,
            "end_sample": end,
        })

    return pd.DataFrame(rows)


def segment_record(
    record_path,
    loader,
    lowcut=0.5,
    highcut=40.0,
    order=4,
    window_sec=2.0,
    step_sec=0.5,
):
    record_path = Path(record_path)
    metadata = read_metadata(record_path)
    processed = preprocess_record(
        record_path=record_path,
        loader=loader,
        lowcut=lowcut,
        highcut=highcut,
        order=order,
        window_sec=window_sec,
        step_sec=step_sec,
    )

    segment_table = build_segment_metadata(
        patient_id=metadata["patient_id"],
        label=metadata["label"],
        segment_ranges=processed["segment_ranges"],
    )

    return {
        "patient_id": metadata["patient_id"],
        "label": metadata["label"],
        "fs": processed["fs"],
        "leads": processed["leads"],
        "checks": processed["checks"],
        "signal": processed["signal"],
        "filtered_signal": processed["filtered_signal"],
        "normalized_signal": processed["normalized_signal"],
        "segments": processed["segments"],
        "segment_ranges": processed["segment_ranges"],
        "segment_table": segment_table,
    }


def segment_dataset(
    records,
    loader,
    lowcut=0.5,
    highcut=40.0,
    order=4,
    window_sec=2.0,
    step_sec=0.5,
):
    patient_segments = {}
    all_segments = []
    all_tables = []
    errors = []

    for record in records:
        try:
            result = segment_record(
                record_path=record.with_suffix(""),
                loader=loader,
                lowcut=lowcut,
                highcut=highcut,
                order=order,
                window_sec=window_sec,
                step_sec=step_sec,
            )

            patient_segments[result["patient_id"]] = result["segments"]

            if len(result["segments"]) > 0:
                all_segments.append(result["segments"])
                all_tables.append(result["segment_table"])

        except Exception as exc:
            errors.append({
                "patient_id": record.stem,
                "error": str(exc),
            })

    if all_segments:
        segments_array = np.concatenate(all_segments, axis=0)
    else:
        segments_array = np.empty((0, 0, 0), dtype=float)

    if all_tables:
        segments_df = pd.concat(all_tables, ignore_index=True)
    else:
        segments_df = pd.DataFrame(
            columns=["patient_id", "segment_id", "label", "segment_ref", "start_sample", "end_sample"]
        )

    errors_df = pd.DataFrame(errors)
    return patient_segments, segments_array, segments_df, errors_df
