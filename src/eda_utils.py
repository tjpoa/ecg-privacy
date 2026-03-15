import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from concurrent.futures import ThreadPoolExecutor

from data_loading import summarize_record_fast


def parse_demographic_metadata(hea_path):
    age = None
    sex = None
    dx = None

    with open(hea_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line.startswith("#Age:"):
                value = line.split(":", 1)[1].strip()
                age = None if value.lower() == "nan" else value

            elif line.startswith("#Sex:"):
                sex = line.split(":", 1)[1].strip()

            elif line.startswith("#Dx:"):
                dx = line.split(":", 1)[1].strip()

    return {
        "record_id": hea_path.stem,
        "age": age,
        "sex": sex,
        "dx": dx,
    }


def _build_metadata_row(hea, loader=None, use_fast_path=True):
    record = hea.with_suffix("")
    row = {"record_id": hea.stem}

    if use_fast_path:
        row.update(summarize_record_fast(record))
    else:
        signal, fs, leads = loader(record)
        lead_ii = signal[:, 1] if signal.shape[1] > 1 else signal[:, 0]
        row.update({
            "samples": signal.shape[0],
            "n_leads": signal.shape[1],
            "fs": fs,
            "duration_sec": signal.shape[0] / fs,
            "global_mean": float(signal.mean()),
            "global_std": float(signal.std()),
            "global_min": float(signal.min()),
            "global_max": float(signal.max()),
            "lead_ii_mean": float(lead_ii.mean()),
            "lead_ii_std": float(lead_ii.std()),
            "lead_ii_min": float(lead_ii.min()),
            "lead_ii_max": float(lead_ii.max()),
            "lead_names": tuple(leads),
        })

    return row


def build_metadata_table(records, loader=None, use_fast_path=True, max_workers=1):
    metadata = []
    errors = []

    def process_record(hea):
        try:
            return _build_metadata_row(hea, loader=loader, use_fast_path=use_fast_path), None
        except Exception as exc:
            return None, {
                "record_id": hea.stem,
                "error": str(exc),
            }

    if max_workers and max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for row, error in executor.map(process_record, records):
                if row is not None:
                    metadata.append(row)
                if error is not None:
                    errors.append(error)
    else:
        for hea in records:
            row, error = process_record(hea)
            if row is not None:
                metadata.append(row)
            if error is not None:
                errors.append(error)

    metadata_df = pd.DataFrame(metadata)
    errors_df = pd.DataFrame(errors)

    return metadata_df, errors_df


def build_demographic_table(records):
    metadata = []
    errors = []

    for hea in records:
        try:
            metadata.append(parse_demographic_metadata(hea))

        except Exception as exc:
            errors.append({
                "record_id": hea.stem,
                "error": str(exc),
            })

    demo_df = pd.DataFrame(metadata)
    errors_df = pd.DataFrame(errors)

    return demo_df, errors_df


def add_afl_label(demo_df, afl_code="164889003"):
    labeled = demo_df.copy()
    dx_series = labeled["dx"].fillna("").astype(str)
    dx_codes = dx_series.str.split(",").apply(lambda codes: {code.strip() for code in codes if code.strip()})
    labeled["is_afl"] = dx_codes.apply(lambda codes: afl_code in codes)
    labeled["label"] = np.where(labeled["is_afl"], "AFL", "Non-AFL")
    return labeled


def plot_random_ecg(records, loader, n=5, lead_idx=1, random_state=42):
    random.seed(random_state)
    sampled = random.sample(records, min(n, len(records)))

    for hea in sampled:
        record = hea.with_suffix("")
        signal, fs, leads = loader(record)

        plt.figure(figsize=(10, 4))
        plt.plot(signal[:, lead_idx])
        plt.title(f"{hea.stem} - Lead {leads[lead_idx]}")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.show()


def segment_signal_fixed(signal, fs, window_sec=2.0, step_sec=1.0):
    window_size = int(window_sec * fs)
    step_size = int(step_sec * fs)

    segments = []
    for start in range(0, signal.shape[0] - window_size + 1, step_size):
        end = start + window_size
        segments.append(signal[start:end])

    return segments


def flatten_segment(segment, lead_idx=1):
    return segment[:, lead_idx]


def zscore_signal(x):
    x = np.asarray(x, dtype=float)
    std = x.std()

    if std == 0:
        return np.zeros_like(x)

    return (x - x.mean()) / std


def cosine_similarity_1d(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return np.nan

    return float(np.dot(x, y) / denom)


def compute_intra_inter_similarity(records, loader, n_records=50, lead_idx=1,
                                   window_sec=2.0, step_sec=1.0, random_state=42,
                                   normalize=True):
    random.seed(random_state)

    sampled_records = random.sample(records, min(n_records, len(records)))
    segmented_records = []
    errors = []

    for hea in sampled_records:
        try:
            record_path = hea.with_suffix("")
            signal, fs, leads = loader(record_path)

            segments = segment_signal_fixed(
                signal=signal,
                fs=fs,
                window_sec=window_sec,
                step_sec=step_sec
            )

            if len(segments) >= 2:
                segmented_records.append({
                    "record_id": hea.stem,
                    "segments": segments
                })

        except Exception as exc:
            errors.append({
                "record_id": hea.stem,
                "error": str(exc),
            })

    intra_scores = []
    inter_scores = []

    for rec in segmented_records:
        seg1, seg2 = random.sample(rec["segments"], 2)
        x = flatten_segment(seg1, lead_idx=lead_idx)
        y = flatten_segment(seg2, lead_idx=lead_idx)
        if normalize:
            x = zscore_signal(x)
            y = zscore_signal(y)

        sim = cosine_similarity_1d(x, y)
        if not np.isnan(sim):
            intra_scores.append(sim)

    for _ in range(len(intra_scores)):
        rec_a, rec_b = random.sample(segmented_records, 2)
        seg_a = random.choice(rec_a["segments"])
        seg_b = random.choice(rec_b["segments"])

        x = flatten_segment(seg_a, lead_idx=lead_idx)
        y = flatten_segment(seg_b, lead_idx=lead_idx)
        if normalize:
            x = zscore_signal(x)
            y = zscore_signal(y)

        sim = cosine_similarity_1d(x, y)
        if not np.isnan(sim):
            inter_scores.append(sim)

    df_intra = pd.DataFrame({
        "similarity": intra_scores,
        "pair_type": "intra-subject"
    })

    df_inter = pd.DataFrame({
        "similarity": inter_scores,
        "pair_type": "inter-subject"
    })

    similarity_df = pd.concat([df_intra, df_inter], ignore_index=True)
    errors_df = pd.DataFrame(errors)

    return similarity_df, errors_df


def build_missing_summary(named_frames):
    summaries = {}

    for name, df in named_frames.items():
        miss = df.isna().sum()
        miss_pct = (miss / len(df) * 100).round(3) if len(df) else pd.Series(dtype=float)

        summary = pd.DataFrame({
            "missing_count": miss,
            "missing_pct": miss_pct
        }).sort_values("missing_count", ascending=False)

        summaries[name] = summary

    return summaries


def plot_similarity_distributions(similarity_df):
    intra = similarity_df[similarity_df["pair_type"] == "intra-subject"]["similarity"]
    inter = similarity_df[similarity_df["pair_type"] == "inter-subject"]["similarity"]

    plt.figure(figsize=(10, 5))
    plt.hist(intra, bins=30, alpha=0.6, label="Intra-subject")
    plt.hist(inter, bins=30, alpha=0.6, label="Inter-subject")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Frequency")
    plt.title("Intra-subject vs inter-subject ECG similarity")
    plt.legend()
    plt.show()
