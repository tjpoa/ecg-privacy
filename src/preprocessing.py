import numpy as np
from scipy.signal import butter, filtfilt


def validate_signal(signal, fs, leads, expected_fs=500.0, expected_duration_sec=10.0, expected_n_leads=12):
    signal = np.asarray(signal, dtype=float)

    if signal.ndim != 2:
        raise ValueError("signal must have shape (samples, leads).")

    n_samples, n_leads = signal.shape
    duration_sec = n_samples / fs

    checks = {
        "fs": fs,
        "duration_sec": duration_sec,
        "n_samples": n_samples,
        "n_leads": n_leads,
        "expected_fs": expected_fs,
        "expected_duration_sec": expected_duration_sec,
        "expected_n_leads": expected_n_leads,
        "fs_ok": np.isclose(fs, expected_fs),
        "duration_ok": np.isclose(duration_sec, expected_duration_sec),
        "n_leads_ok": n_leads == expected_n_leads,
        "has_nan": bool(np.isnan(signal).any()),
        "has_inf": bool(np.isinf(signal).any()),
        "has_invalid_values": bool(np.isnan(signal).any() or np.isinf(signal).any()),
        "lead_names_match": len(leads) == n_leads,
        "is_valid": True,
    }
    checks["is_valid"] = (
        checks["fs_ok"]
        and checks["duration_ok"]
        and checks["n_leads_ok"]
        and checks["lead_names_match"]
        and not checks["has_invalid_values"]
    )

    return checks


def butter_bandpass(lowcut=0.5, highcut=40.0, fs=500.0, order=4):
    nyquist = 0.5 * fs

    if lowcut <= 0:
        raise ValueError("lowcut must be > 0.")
    if highcut >= nyquist:
        raise ValueError("highcut must be < Nyquist frequency.")
    if lowcut >= highcut:
        raise ValueError("lowcut must be smaller than highcut.")

    return butter(order, [lowcut / nyquist, highcut / nyquist], btype="band")


def bandpass_filter(signal, fs, lowcut=0.5, highcut=40.0, order=4):
    signal = np.asarray(signal, dtype=float)

    if signal.ndim != 2:
        raise ValueError("signal must have shape (samples, leads).")

    b, a = butter_bandpass(lowcut=lowcut, highcut=highcut, fs=fs, order=order)
    return filtfilt(b, a, signal, axis=0)


def zscore_normalize(signal):
    signal = np.asarray(signal, dtype=float)

    if signal.ndim != 2:
        raise ValueError("signal must have shape (samples, leads).")

    mean = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)

    normalized_signal = (signal - mean) / std
    return normalized_signal, mean.squeeze(axis=0), std.squeeze(axis=0)


def segment_signal(signal, fs, window_sec=2.0, overlap=0.75, step_sec=None):
    signal = np.asarray(signal)

    if signal.ndim != 2:
        raise ValueError("signal must have shape (samples, leads).")

    window_samples = int(window_sec * fs)

    if step_sec is not None:
        step_samples = int(step_sec * fs)
    else:
        if not (0 <= overlap < 1):
            raise ValueError("overlap must be in the interval [0, 1).")
        step_samples = int(window_samples * (1 - overlap))

    if window_samples <= 0:
        raise ValueError("window_sec must produce at least one sample.")
    if step_samples <= 0:
        raise ValueError("step size must be at least one sample.")

    if signal.shape[0] < window_samples:
        return np.empty((0, window_samples, signal.shape[1]), dtype=signal.dtype)

    segments = []
    segment_ranges = []

    for start in range(0, signal.shape[0] - window_samples + 1, step_samples):
        end = start + window_samples
        segments.append(signal[start:end, :])
        segment_ranges.append((start, end))

    return np.stack(segments, axis=0), segment_ranges


def preprocess_signal(
    signal,
    fs,
    leads,
    lowcut=0.5,
    highcut=40.0,
    order=4,
    window_sec=2.0,
    step_sec=0.5,
):
    checks = validate_signal(signal, fs, leads)
    filtered_signal = bandpass_filter(
        signal,
        fs=fs,
        lowcut=lowcut,
        highcut=highcut,
        order=order,
    )
    normalized_signal, lead_means, lead_stds = zscore_normalize(filtered_signal)
    segments, segment_ranges = segment_signal(
        normalized_signal,
        fs=fs,
        window_sec=window_sec,
        step_sec=step_sec,
    )

    return {
        "checks": checks,
        "filtered_signal": filtered_signal,
        "normalized_signal": normalized_signal,
        "segments": segments,
        "segment_ranges": segment_ranges,
        "lead_means": lead_means,
        "lead_stds": lead_stds,
    }


def preprocess_record(
    record_path,
    loader,
    lowcut=0.5,
    highcut=40.0,
    order=4,
    window_sec=2.0,
    step_sec=0.5,
):
    signal, fs, leads = loader(record_path)
    processed = preprocess_signal(
        signal,
        fs=fs,
        leads=leads,
        lowcut=lowcut,
        highcut=highcut,
        order=order,
        window_sec=window_sec,
        step_sec=step_sec,
    )

    processed.update({
        "signal": signal,
        "fs": fs,
        "leads": leads,
    })
    return processed
