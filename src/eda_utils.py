import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


def build_metadata_table(records, loader):
    metadata = []

    for hea in records:
        try:
            record = hea.with_suffix("")
            signal, fs, leads = loader(record)

            metadata.append({
                "record_id": hea.stem,
                "samples": signal.shape[0],
                "leads": signal.shape[1],
                "fs": fs,
                "duration_sec": signal.shape[0] / fs,
                "mean": float(signal.mean()),
                "std": float(signal.std()),
                "min": float(signal.min()),
                "max": float(signal.max())
            })

        except Exception:
            continue

    return pd.DataFrame(metadata)


def build_demographic_table(records):
    metadata = []

    for hea in records:
        try:
            age = None
            sex = None
            dx = None

            with open(hea, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()

                    if line.startswith("#Age:"):
                        value = line.split(":", 1)[1].strip()
                        age = None if value.lower() == "nan" else value

                    elif line.startswith("#Sex:"):
                        sex = line.split(":", 1)[1].strip()

                    elif line.startswith("#Dx:"):
                        dx = line.split(":", 1)[1].strip()

            metadata.append({
                "record_id": hea.stem,
                "age": age,
                "sex": sex,
                "dx": dx
            })

        except Exception:
            continue

    return pd.DataFrame(metadata)


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


def cosine_similarity_1d(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return np.nan

    return float(np.dot(x, y) / denom)


def compute_intra_inter_similarity(records, loader, n_records=50, lead_idx=1,
                                   window_sec=2.0, step_sec=1.0, random_state=42):
    random.seed(random_state)

    sampled_records = random.sample(records, min(n_records, len(records)))
    segmented_records = []

    for hea in sampled_records:
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

    intra_scores = []
    inter_scores = []

    for rec in segmented_records:
        seg1, seg2 = random.sample(rec["segments"], 2)
        x = flatten_segment(seg1, lead_idx=lead_idx)
        y = flatten_segment(seg2, lead_idx=lead_idx)

        sim = cosine_similarity_1d(x, y)
        if not np.isnan(sim):
            intra_scores.append(sim)

    for _ in range(len(intra_scores)):
        rec_a, rec_b = random.sample(segmented_records, 2)
        seg_a = random.choice(rec_a["segments"])
        seg_b = random.choice(rec_b["segments"])

        x = flatten_segment(seg_a, lead_idx=lead_idx)
        y = flatten_segment(seg_b, lead_idx=lead_idx)

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

    return pd.concat([df_intra, df_inter], ignore_index=True)


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