import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew


def map_clinical_label(
    dx_label,
    positive_codes=None,
    positive_label="AFL",
    negative_label="Non-AFL",
):
    if dx_label is None or (isinstance(dx_label, float) and np.isnan(dx_label)):
        return negative_label

    if positive_codes is None:
        positive_codes = {"164889003"}
    elif isinstance(positive_codes, str):
        positive_codes = {positive_codes}
    else:
        positive_codes = {str(code).strip() for code in positive_codes if str(code).strip()}

    codes = {code.strip() for code in str(dx_label).split(",") if code.strip()}
    return positive_label if codes.intersection(positive_codes) else negative_label


def zero_crossing_rate(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return 0.0

    signs = np.signbit(x)
    return float(np.mean(signs[:-1] != signs[1:]))


def detect_simple_peaks(x, fs, min_distance_sec=0.2, prominence_scale=0.5):
    x = np.asarray(x, dtype=float)

    if len(x) < 3:
        return np.array([], dtype=int)

    distance = max(1, int(min_distance_sec * fs))
    prominence = max(0.1, prominence_scale * float(np.std(x)))
    height = float(np.percentile(x, 75))

    peaks, _ = find_peaks(x, distance=distance, prominence=prominence, height=height)
    return peaks


def compute_statistical_features(x):
    x = np.asarray(x, dtype=float)

    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "amplitude": float(np.max(x) - np.min(x)),
        "energy": float(np.sum(np.square(x))),
        "skewness": float(skew(x, bias=False)) if np.std(x) > 0 else 0.0,
        "kurtosis": float(kurtosis(x, fisher=True, bias=False)) if np.std(x) > 0 else 0.0,
    }


def compute_morphological_features(x, fs):
    x = np.asarray(x, dtype=float)
    peaks = detect_simple_peaks(x, fs=fs)

    return {
        "area": float(np.trapezoid(x)),
        "abs_mean": float(np.mean(np.abs(x))),
        "rms": float(np.sqrt(np.mean(np.square(x)))),
        "zero_crossing_rate": zero_crossing_rate(x),
        "n_peaks": int(len(peaks)),
    }


def detect_r_peaks(segment, leads, fs, preferred_lead="II"):
    if preferred_lead in leads:
        lead_idx = leads.index(preferred_lead)
    else:
        lead_idx = 0

    lead_signal = np.asarray(segment[:, lead_idx], dtype=float)
    return detect_simple_peaks(lead_signal, fs=fs, min_distance_sec=0.25, prominence_scale=0.6)


def compute_rr_features(r_peaks, fs):
    if len(r_peaks) < 2:
        return {
            "mean_rr": np.nan,
            "std_rr": np.nan,
            "rmssd": np.nan,
            "mean_hr_bpm": np.nan,
            "n_r_peaks": int(len(r_peaks)),
        }

    rr_intervals = np.diff(r_peaks) / fs
    rr_diffs = np.diff(rr_intervals)
    mean_rr = float(np.mean(rr_intervals))

    return {
        "mean_rr": mean_rr,
        "std_rr": float(np.std(rr_intervals)),
        "rmssd": float(np.sqrt(np.mean(np.square(rr_diffs)))) if len(rr_diffs) > 0 else np.nan,
        "mean_hr_bpm": float(60.0 / mean_rr) if mean_rr > 0 else np.nan,
        "n_r_peaks": int(len(r_peaks)),
    }


def compute_lead_features(segment_1d, fs):
    x = np.asarray(segment_1d, dtype=float)
    features = {}
    features.update(compute_statistical_features(x))
    features.update(compute_morphological_features(x, fs=fs))
    return features


def aggregate_feature_group(per_lead_features):
    feature_names = per_lead_features[0].keys()
    aggregated = {}

    for feature_name in feature_names:
        values = np.asarray([row[feature_name] for row in per_lead_features], dtype=float)
        aggregated[f"global_mean_{feature_name}"] = float(np.mean(values))
        aggregated[f"global_std_{feature_name}"] = float(np.std(values))
        aggregated[f"global_min_{feature_name}"] = float(np.min(values))
        aggregated[f"global_max_{feature_name}"] = float(np.max(values))

    return aggregated


def extract_segment_features(segment, leads, fs, include_rr_features=False, rr_lead="II"):
    segment = np.asarray(segment, dtype=float)

    if segment.ndim != 2:
        raise ValueError("segment must have shape (window_samples, leads).")
    if len(leads) != segment.shape[1]:
        raise ValueError("number of lead names must match segment.shape[1].")

    feature_row = {}
    per_lead_rows = []

    for lead_idx, lead_name in enumerate(leads):
        lead_features = compute_lead_features(segment[:, lead_idx], fs=fs)
        per_lead_rows.append(lead_features)

        safe_lead_name = str(lead_name).replace(" ", "_")
        for feature_name, value in lead_features.items():
            feature_row[f"lead_{safe_lead_name}_{feature_name}"] = value

    feature_row.update(aggregate_feature_group(per_lead_rows))

    if include_rr_features:
        r_peaks = detect_r_peaks(segment, leads=leads, fs=fs, preferred_lead=rr_lead)
        rr_features = compute_rr_features(r_peaks, fs=fs)
        for feature_name, value in rr_features.items():
            feature_row[f"rhythm_{feature_name}"] = value

    return feature_row


def build_features_dataframe(
    segments_array,
    segments_df,
    leads,
    fs,
    afl_code="164889003",
    utility_codes=None,
    positive_label="AFL",
    negative_label="Non-AFL",
    include_rr_features=False,
    rr_lead="II",
):
    segments_array = np.asarray(segments_array)

    if len(segments_array) != len(segments_df):
        raise ValueError("segments_array and segments_df must have the same number of segments.")

    rows = []
    for idx, (_, meta_row) in enumerate(segments_df.reset_index(drop=True).iterrows()):
        row = {
            "patient_id": meta_row["patient_id"],
            "segment_id": meta_row["segment_id"],
            "label": meta_row["label"],
            "utility_label": map_clinical_label(
                meta_row["label"],
                positive_codes=utility_codes or {afl_code},
                positive_label=positive_label,
                negative_label=negative_label,
            ),
            "segment_ref": meta_row["segment_ref"],
            "start_sample": meta_row["start_sample"],
            "end_sample": meta_row["end_sample"],
        }
        row.update(
            extract_segment_features(
                segments_array[idx],
                leads=leads,
                fs=fs,
                include_rr_features=include_rr_features,
                rr_lead=rr_lead,
            )
        )
        rows.append(row)

    return pd.DataFrame(rows)


def build_record_features_dataset(
    records,
    loader,
    lowcut=0.5,
    highcut=40.0,
    order=4,
    window_sec=10.0,
    step_sec=10.0,
    utility_codes=None,
    positive_label="Atrial",
    negative_label="Non-Atrial",
    include_rr_features=True,
    rr_lead="II",
    progress_every=None,
):
    from preprocessing import preprocess_record
    from segmentation import read_metadata

    rows = []
    errors = []

    total_records = len(records)

    for idx, record_path in enumerate(records, start=1):
        try:
            metadata = read_metadata(record_path)
            processed = preprocess_record(
                record_path.with_suffix(""),
                loader=loader,
                lowcut=lowcut,
                highcut=highcut,
                order=order,
                window_sec=window_sec,
                step_sec=step_sec,
            )

            signal_features = extract_segment_features(
                processed["normalized_signal"],
                leads=processed["leads"],
                fs=processed["fs"],
                include_rr_features=include_rr_features,
                rr_lead=rr_lead,
            )

            rows.append(
                {
                    "patient_id": metadata["patient_id"],
                    "record_id": metadata["patient_id"],
                    "label": metadata["label"],
                    "utility_label": map_clinical_label(
                        metadata["label"],
                        positive_codes=utility_codes or {"164889003", "164890007"},
                        positive_label=positive_label,
                        negative_label=negative_label,
                    ),
                    **signal_features,
                }
            )
        except Exception as exc:
            errors.append(
                {
                    "record_path": str(record_path),
                    "error": str(exc),
                }
            )

        if progress_every and idx % progress_every == 0:
            print(f"Processed {idx}/{total_records} records")

    return pd.DataFrame(rows), pd.DataFrame(errors)


def summarize_features(features_df):
    numeric_df = features_df.select_dtypes(include=[np.number])

    return {
        "n_rows": len(features_df),
        "n_columns": features_df.shape[1],
        "n_numeric_features": numeric_df.shape[1],
        "missing_values": int(features_df.isna().sum().sum()),
        "utility_label_counts": features_df["utility_label"].value_counts(dropna=False).to_dict()
        if "utility_label" in features_df.columns
        else {},
    }
