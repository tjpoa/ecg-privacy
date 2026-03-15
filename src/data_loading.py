from pathlib import Path

import numpy as np
import wfdb
from scipy.io import loadmat


def parse_header(record_path):
    hea_path = Path(record_path).with_suffix(".hea")
    with open(hea_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    first_parts = lines[0].split()
    n_leads = int(first_parts[1])
    fs = float(first_parts[2])

    if len(first_parts) > 3 and "/" not in first_parts[3]:
        n_samples = int(first_parts[3])
        signal_lines = lines[1:1 + n_leads]
    else:
        n_samples = None
        signal_lines = lines[:n_leads]

    lead_names = []
    adc_gains = []
    baselines = []

    for line in signal_lines:
        parts = line.split()
        gain_field = parts[2]

        if "/" in gain_field:
            gain_value = gain_field.split("/", 1)[0]
        else:
            gain_value = gain_field

        if "(" in gain_value:
            gain_str, baseline_str = gain_value.split("(", 1)
            baseline = float(baseline_str.rstrip(")"))
        else:
            gain_str = gain_value
            baseline = float(parts[4]) if len(parts) > 4 else 0.0

        gain = float(gain_str) if gain_str not in {"0", ""} else 1.0

        adc_gains.append(gain)
        baselines.append(baseline)
        lead_names.append(parts[-1])

    return {
        "fs": fs,
        "n_samples": n_samples,
        "n_leads": n_leads,
        "lead_names": lead_names,
        "adc_gains": np.asarray(adc_gains, dtype=float),
        "baselines": np.asarray(baselines, dtype=float),
    }


def load_signal_fast(record_path, physical=True):
    record_path = Path(record_path)
    signal = loadmat(record_path.with_suffix(".mat"))["val"].T.astype(np.float32, copy=False)
    header = parse_header(record_path)
    if header["n_samples"] is None:
        header["n_samples"] = signal.shape[0]

    if physical:
        signal = (signal - header["baselines"]) / header["adc_gains"]

    return signal, header["fs"], header["lead_names"]


def summarize_record_fast(record_path):
    signal, fs, leads = load_signal_fast(record_path, physical=True)
    lead_ii = signal[:, 1] if signal.shape[1] > 1 else signal[:, 0]

    return {
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
    }

def load_record(record_path):
    """
    Load ECG record from WFDB format.
    """
    
    record = wfdb.rdrecord(str(record_path))

    signal = record.p_signal
    fs = record.fs
    leads = record.sig_name

    return signal, fs, leads
