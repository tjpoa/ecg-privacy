from __future__ import annotations

import argparse
import json
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

try:
    from config import (
        DATA_INTERIM,
        FINAL_SEGMENT_FEATURES_DIR,
        OUTPUTS_FIGURES_DIR,
        OUTPUTS_TABLES_DIR,
        WFDB_RECORDS_DIR,
    )
    from data_loading import load_signal_fast
    from modeling import get_feature_columns, split_segments_by_patient_stratified
    from preprocessing import preprocess_record
    from run_encoder_dp_study import apply_dp_to_embedding_split
    from run_operational_linkage_eval import load_feature_dataset
    from run_representation_dimension_comparison import build_supervised_encoder_representation
except ImportError:  # pragma: no cover - package import fallback
    from .config import (
        DATA_INTERIM,
        FINAL_SEGMENT_FEATURES_DIR,
        OUTPUTS_FIGURES_DIR,
        OUTPUTS_TABLES_DIR,
        WFDB_RECORDS_DIR,
    )
    from .data_loading import load_signal_fast
    from .modeling import get_feature_columns, split_segments_by_patient_stratified
    from .preprocessing import preprocess_record
    from .run_encoder_dp_study import apply_dp_to_embedding_split
    from .run_operational_linkage_eval import load_feature_dataset
    from .run_representation_dimension_comparison import build_supervised_encoder_representation


STANDARD_12_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train empirical waveform-reconstruction attacks against raw handcrafted features, "
            "utility bottleneck embeddings, and Gaussian-noised embeddings."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, default=FINAL_SEGMENT_FEATURES_DIR)
    parser.add_argument("--records-dir", type=Path, default=WFDB_RECORDS_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_TABLES_DIR)
    parser.add_argument("--figure-dir", type=Path, default=OUTPUTS_FIGURES_DIR)
    parser.add_argument("--target-cache", type=Path, default=None)
    parser.add_argument("--run-name", default="reconstruction_attack")
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=20,
        help="Number of feature chunks to load. Use 0 for the complete dataset.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--target-records", type=int, default=12000)
    parser.add_argument("--cohort-seed", type=int, default=2026)
    parser.add_argument("--segment-ref", type=int, default=0)
    parser.add_argument("--target-fs", type=int, default=100)
    parser.add_argument("--target-pca-components", type=int, default=128)
    parser.add_argument("--target-workers", type=int, default=8)
    parser.add_argument("--decoders", nargs="+", choices=["ridge", "mlp"], default=["ridge", "mlp"])
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--decoder-hidden-units", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--decoder-max-iter", type=int, default=100)
    parser.add_argument("--decoder-batch-size", type=int, default=256)
    parser.add_argument("--decoder-learning-rate-init", type=float, default=0.001)
    parser.add_argument("--max-lag-ms", type=float, default=250.0)
    parser.add_argument("--bottleneck-dim", type=int, default=8)
    parser.add_argument("--encoder-hidden-units", type=int, default=128)
    parser.add_argument("--encoder-max-iter", type=int, default=40)
    parser.add_argument("--encoder-batch-size", type=int, default=1024)
    parser.add_argument("--encoder-learning-rate-init", type=float, default=0.001)
    parser.add_argument("--encoder-train-max-segments", type=int, default=80000)
    parser.add_argument("--dp-epsilon", type=float, default=50.0)
    parser.add_argument("--dp-delta", type=float, default=1e-5)
    parser.add_argument("--dp-clip-norm", type=float, default=2.0)
    return parser.parse_args()


def default_target_cache(args: argparse.Namespace) -> Path:
    return DATA_INTERIM / (
        f"reconstruction_targets_ref{args.segment_ref}_{args.target_fs}hz_n{args.target_records}.npz"
    )


def extract_target_for_record(
    patient_id: str,
    record_path: Path,
    segment_ref: int,
    target_fs: int,
    preprocessing: dict,
) -> tuple[str, np.ndarray | None, str | None]:
    try:
        processed = preprocess_record(
            record_path=record_path.with_suffix(""),
            loader=load_signal_fast,
            lowcut=float(preprocessing["lowcut"]),
            highcut=float(preprocessing["highcut"]),
            order=int(preprocessing["order"]),
            window_sec=float(preprocessing["window_sec"]),
            step_sec=float(preprocessing["step_sec"]),
        )
        segments = processed["segments"]
        if segment_ref < 0 or segment_ref >= len(segments):
            raise IndexError(f"segment_ref {segment_ref} is unavailable")
        segment = np.asarray(segments[segment_ref], dtype=np.float32)
        source_fs = int(round(float(processed["fs"])))
        if source_fs % target_fs == 0:
            downsampled = resample_poly(segment, up=1, down=source_fs // target_fs, axis=0)
        else:
            downsampled = resample_poly(segment, up=target_fs, down=source_fs, axis=0)
        # Lead-major flattening makes per-lead metric reshaping explicit.
        flattened = np.asarray(downsampled.T.reshape(-1), dtype=np.float32)
        return patient_id, flattened, None
    except Exception as exc:  # pragma: no cover - defensive batch extraction
        return patient_id, None, str(exc)


def build_or_load_target_cohort(
    features_df: pd.DataFrame,
    manifest: dict,
    records_dir: Path,
    cache_path: Path,
    target_records: int,
    cohort_seed: int,
    segment_ref: int,
    target_fs: int,
    workers: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        patient_ids = cached["patient_ids"].astype(str)
        waveforms = cached["waveforms"].astype(np.float32, copy=False)
        metadata = {
            "cache_path": str(cache_path),
            "cache_reused": True,
            "requested_records": int(target_records),
            "retained_records": int(len(patient_ids)),
            "errors": int(cached["error_count"][0]) if "error_count" in cached else 0,
        }
        return patient_ids, waveforms, metadata

    available_ids = (
        features_df.loc[features_df["segment_ref"].astype(int) == int(segment_ref), "patient_id"]
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .to_numpy(dtype=object)
    )
    rng = np.random.default_rng(cohort_seed)
    n_select = min(int(target_records), len(available_ids)) if target_records > 0 else len(available_ids)
    selected_ids = rng.choice(available_ids, size=n_select, replace=False)
    selected_set = set(str(value) for value in selected_ids)

    record_paths = {
        path.stem: path
        for path in records_dir.rglob("*.hea")
        if path.stem in selected_set
    }
    tasks = [(str(patient_id), record_paths.get(str(patient_id))) for patient_id in selected_ids]
    missing_paths = [patient_id for patient_id, path in tasks if path is None]
    tasks = [(patient_id, path) for patient_id, path in tasks if path is not None]
    preprocessing = {
        "lowcut": manifest.get("lowcut", 0.5),
        "highcut": manifest.get("highcut", 40.0),
        "order": manifest.get("order", 4),
        "window_sec": manifest.get("window_sec", 2.0),
        "step_sec": manifest.get("step_sec", 1.0),
    }

    def process(task):
        patient_id, path = task
        return extract_target_for_record(
            patient_id=patient_id,
            record_path=path,
            segment_ref=segment_ref,
            target_fs=target_fs,
            preprocessing=preprocessing,
        )

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(process, tasks))
    else:
        results = [process(task) for task in tasks]

    retained_ids = []
    retained_waveforms = []
    errors = [{"patient_id": value, "error": "header not found"} for value in missing_paths]
    for patient_id, waveform, error in results:
        if waveform is None:
            errors.append({"patient_id": patient_id, "error": error})
        else:
            retained_ids.append(patient_id)
            retained_waveforms.append(waveform)
    if not retained_waveforms:
        raise ValueError("No waveform targets could be extracted.")

    patient_ids = np.asarray(retained_ids, dtype=str)
    waveforms = np.stack(retained_waveforms).astype(np.float32, copy=False)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        patient_ids=patient_ids,
        waveforms=waveforms,
        segment_ref=np.asarray([segment_ref], dtype=int),
        target_fs=np.asarray([target_fs], dtype=int),
        error_count=np.asarray([len(errors)], dtype=int),
    )
    if errors:
        pd.DataFrame(errors).to_csv(cache_path.with_suffix(".errors.csv"), index=False)

    metadata = {
        "cache_path": str(cache_path),
        "cache_reused": False,
        "requested_records": int(target_records),
        "retained_records": int(len(patient_ids)),
        "errors": int(len(errors)),
        "preprocessing": preprocessing,
    }
    return patient_ids, waveforms, metadata


def select_representation_rows(
    representation_df: pd.DataFrame,
    cohort_ids: set[str],
    segment_ref: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feature_columns = get_feature_columns(representation_df)
    selected = representation_df[
        (representation_df["segment_ref"].astype(int) == int(segment_ref))
        & representation_df["patient_id"].astype(str).isin(cohort_ids)
    ].copy()
    selected = selected.drop_duplicates("patient_id", keep="first").sort_values("patient_id")
    patient_ids = selected["patient_id"].astype(str).to_numpy()
    matrix = selected[feature_columns].to_numpy(dtype=np.float32)
    matrix[~np.isfinite(matrix)] = np.nan
    return patient_ids, matrix, feature_columns


def align_targets(patient_ids: np.ndarray, target_ids: np.ndarray, target_matrix: np.ndarray) -> np.ndarray:
    target_positions = {str(patient_id): index for index, patient_id in enumerate(target_ids)}
    positions = [target_positions[str(patient_id)] for patient_id in patient_ids]
    return target_matrix[np.asarray(positions, dtype=int)]


def rowwise_correlations(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    true_centered = y_true - y_true.mean(axis=-1, keepdims=True)
    pred_centered = y_pred - y_pred.mean(axis=-1, keepdims=True)
    numerator = np.sum(true_centered * pred_centered, axis=-1)
    denominator = np.linalg.norm(true_centered, axis=-1) * np.linalg.norm(pred_centered, axis=-1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 1e-12,
    )


def max_lag_correlation(y_true: np.ndarray, y_pred: np.ndarray, max_lag: int) -> np.ndarray:
    scores = np.full(len(y_true), -1.0, dtype=float)
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            current = rowwise_correlations(y_true[:, :lag], y_pred[:, -lag:])
        elif lag > 0:
            current = rowwise_correlations(y_true[:, lag:], y_pred[:, :-lag])
        else:
            current = rowwise_correlations(y_true, y_pred)
        scores = np.maximum(scores, current)
    return scores


def reconstruction_metrics(
    patient_ids: np.ndarray,
    y_true_flat: np.ndarray,
    y_pred_flat: np.ndarray,
    n_leads: int,
    target_fs: int,
    max_lag_ms: float,
) -> tuple[dict, pd.DataFrame]:
    samples_per_lead = y_true_flat.shape[1] // n_leads
    y_true = y_true_flat.reshape(-1, n_leads, samples_per_lead)
    y_pred = y_pred_flat.reshape(-1, n_leads, samples_per_lead)
    error = y_true - y_pred

    rmse = np.sqrt(np.mean(error**2, axis=(1, 2)))
    true_norm = np.linalg.norm(y_true.reshape(len(y_true), -1), axis=1)
    error_norm = np.linalg.norm(error.reshape(len(error), -1), axis=1)
    prd = 100.0 * np.divide(error_norm, np.maximum(true_norm, 1e-12))
    snr_db = 20.0 * np.log10(np.maximum(true_norm, 1e-12) / np.maximum(error_norm, 1e-12))

    lead_correlations = rowwise_correlations(y_true, y_pred)
    lead_true_norm = np.linalg.norm(y_true, axis=2)
    lead_error_norm = np.linalg.norm(error, axis=2)
    lead_prd = np.full_like(lead_error_norm, np.nan, dtype=float)
    np.divide(
        100.0 * lead_error_norm,
        lead_true_norm,
        out=lead_prd,
        where=lead_true_norm > 1e-6,
    )
    lead_ii_index = 1 if n_leads > 1 else 0
    lead_ii_corr = lead_correlations[:, lead_ii_index]
    max_lag = int(round(max_lag_ms / 1000.0 * target_fs))
    lead_ii_maxlag = max_lag_correlation(
        y_true[:, lead_ii_index, :],
        y_pred[:, lead_ii_index, :],
        max_lag=max_lag,
    )

    true_spectrum = np.log1p(np.abs(np.fft.rfft(y_true[:, lead_ii_index, :], axis=1)))
    pred_spectrum = np.log1p(np.abs(np.fft.rfft(y_pred[:, lead_ii_index, :], axis=1)))
    lead_ii_spectral_corr = rowwise_correlations(true_spectrum, pred_spectrum)

    per_record = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "rmse": rmse,
            "prd_percent": prd,
            "snr_db": snr_db,
            "mean_lead_correlation": np.mean(lead_correlations, axis=1),
            "lead_ii_correlation": lead_ii_corr,
            "lead_ii_maxlag_correlation": lead_ii_maxlag,
            "lead_ii_spectral_correlation": lead_ii_spectral_corr,
        }
    )
    lead_names = STANDARD_12_LEADS if n_leads == 12 else [f"lead_{index + 1}" for index in range(n_leads)]
    for lead_index, lead_name in enumerate(lead_names):
        per_record[f"lead_{lead_name}_correlation"] = lead_correlations[:, lead_index]
        per_record[f"lead_{lead_name}_prd_percent"] = lead_prd[:, lead_index]
    summary = {
        "n_test": int(len(per_record)),
        "rmse_mean": float(per_record["rmse"].mean()),
        "rmse_median": float(per_record["rmse"].median()),
        "prd_mean": float(per_record["prd_percent"].mean()),
        "prd_median": float(per_record["prd_percent"].median()),
        "snr_db_mean": float(per_record["snr_db"].mean()),
        "mean_lead_correlation_mean": float(per_record["mean_lead_correlation"].mean()),
        "lead_ii_correlation_mean": float(per_record["lead_ii_correlation"].mean()),
        "lead_ii_correlation_median": float(per_record["lead_ii_correlation"].median()),
        "lead_ii_maxlag_correlation_mean": float(per_record["lead_ii_maxlag_correlation"].mean()),
        "lead_ii_spectral_correlation_mean": float(per_record["lead_ii_spectral_correlation"].mean()),
    }
    return summary, per_record


def fit_decoder(
    decoder_name: str,
    X_train: np.ndarray,
    target_components_train: np.ndarray,
    seed: int,
    args: argparse.Namespace,
):
    if decoder_name == "ridge":
        model = Ridge(alpha=args.ridge_alpha)
    elif decoder_name == "mlp":
        effective_batch_size = min(args.decoder_batch_size, max(1, int(0.8 * len(X_train))))
        model = MLPRegressor(
            hidden_layer_sizes=tuple(args.decoder_hidden_units),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=effective_batch_size,
            learning_rate_init=args.decoder_learning_rate_init,
            max_iter=args.decoder_max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=seed,
            verbose=False,
        )
    else:
        raise ValueError(f"Unsupported decoder: {decoder_name}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X_train, target_components_train)
    return model


def evaluate_representation_decoder(
    representation: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_ids: np.ndarray,
    target_matrix: np.ndarray,
    target_pca: PCA,
    target_scaler: StandardScaler,
    seed: int,
    args: argparse.Namespace,
) -> tuple[list[dict], list[pd.DataFrame], dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    cohort_set = set(str(value) for value in target_ids)
    train_ids, X_train, feature_columns = select_representation_rows(
        train_df, cohort_ids=cohort_set, segment_ref=args.segment_ref
    )
    test_ids, X_test, _ = select_representation_rows(
        test_df, cohort_ids=cohort_set, segment_ref=args.segment_ref
    )
    y_train = align_targets(train_ids, target_ids, target_matrix)
    y_test = align_targets(test_ids, target_ids, target_matrix)

    imputer = SimpleImputer(strategy="median")
    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(imputer.fit_transform(X_train))
    X_test = x_scaler.transform(imputer.transform(X_test))
    target_components_train = target_scaler.transform(target_pca.transform(y_train))

    metrics_rows = []
    per_record_frames = []
    examples = {}
    for decoder_name in args.decoders:
        model = fit_decoder(
            decoder_name=decoder_name,
            X_train=X_train,
            target_components_train=target_components_train,
            seed=seed,
            args=args,
        )
        predicted_components = target_scaler.inverse_transform(model.predict(X_test))
        y_pred = target_pca.inverse_transform(predicted_components).astype(np.float32, copy=False)
        metrics, per_record = reconstruction_metrics(
            patient_ids=test_ids,
            y_true_flat=y_test,
            y_pred_flat=y_pred,
            n_leads=12,
            target_fs=args.target_fs,
            max_lag_ms=args.max_lag_ms,
        )
        model_n_iter = getattr(model, "n_iter_", None)
        if model_n_iter is None:
            serialized_n_iter = np.nan
        else:
            n_iter_array = np.asarray(model_n_iter).reshape(-1)
            serialized_n_iter = int(n_iter_array[0]) if len(n_iter_array) else np.nan
        metrics_rows.append(
            {
                "seed": seed,
                "representation": representation,
                "decoder": decoder_name,
                "n_train": int(len(train_ids)),
                "n_features": int(len(feature_columns)),
                "decoder_n_iter": serialized_n_iter,
                **metrics,
            }
        )
        per_record.insert(0, "decoder", decoder_name)
        per_record.insert(0, "representation", representation)
        per_record.insert(0, "seed", seed)
        per_record_frames.append(per_record)
        if decoder_name == "mlp":
            examples[representation] = (test_ids[:3], y_test[:3], y_pred[:3])
    return metrics_rows, per_record_frames, examples


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["representation", "decoder"]
    metric_columns = [
        "n_train",
        "n_test",
        "rmse_mean",
        "prd_mean",
        "prd_median",
        "snr_db_mean",
        "mean_lead_correlation_mean",
        "lead_ii_correlation_mean",
        "lead_ii_maxlag_correlation_mean",
        "lead_ii_spectral_correlation_mean",
    ]
    summary = metrics_df.groupby(group_columns, as_index=False)[metric_columns].agg(["mean", "std"])
    summary.columns = [
        "_".join([part for part in column if part]) if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    return summary


def summarize_per_lead(per_record_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (representation, decoder), group in per_record_df.groupby(["representation", "decoder"], sort=True):
        for lead_name in STANDARD_12_LEADS:
            corr = group[f"lead_{lead_name}_correlation"]
            prd = group[f"lead_{lead_name}_prd_percent"]
            rows.append(
                {
                    "representation": representation,
                    "decoder": decoder,
                    "lead": lead_name,
                    "n_records_across_seeds": int(len(group)),
                    "n_valid_prd_records": int(prd.notna().sum()),
                    "correlation_mean": float(corr.mean()),
                    "correlation_std": float(corr.std()),
                    "correlation_q25": float(corr.quantile(0.25)),
                    "correlation_median": float(corr.median()),
                    "correlation_q75": float(corr.quantile(0.75)),
                    "prd_mean": float(prd.mean()),
                    "prd_std": float(prd.std()),
                    "prd_q25": float(prd.quantile(0.25)),
                    "prd_median": float(prd.median()),
                    "prd_q75": float(prd.quantile(0.75)),
                }
            )
    return pd.DataFrame(rows)


def plot_examples(
    examples: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    target_fs: int,
    output_path: Path,
) -> None:
    representation_order = ["raw_features", "encoder", "encoder_gaussian_noise"]
    available = [name for name in representation_order if name in examples]
    if not available:
        return
    n_examples = min(3, *(len(examples[name][0]) for name in available))
    fig, axes = plt.subplots(n_examples, len(available), figsize=(4.2 * len(available), 2.6 * n_examples), squeeze=False)
    labels = {
        "raw_features": "Raw features",
        "encoder": "Encoder",
        "encoder_gaussian_noise": "Encoder + noise",
    }
    for column, representation in enumerate(available):
        patient_ids, y_true_flat, y_pred_flat = examples[representation]
        samples = y_true_flat.shape[1] // 12
        y_true = y_true_flat.reshape(-1, 12, samples)
        y_pred = y_pred_flat.reshape(-1, 12, samples)
        time_axis = np.arange(samples) / target_fs
        for row in range(n_examples):
            ax = axes[row, column]
            ax.plot(time_axis, y_true[row, 1], color="black", linewidth=1.0, label="True")
            ax.plot(time_axis, y_pred[row, 1], color="#D95F02", linewidth=1.0, alpha=0.9, label="Reconstructed")
            ax.set_title(f"{labels[representation]} — {patient_ids[row]}", fontsize=9)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Normalized Lead II")
            ax.grid(alpha=0.2)
            if row == 0 and column == 0:
                ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    max_chunks = None if args.max_chunks == 0 else args.max_chunks
    features_df, manifest, chunk_files = load_feature_dataset(args.dataset_dir, max_chunks=max_chunks)
    features_df["patient_id"] = features_df["patient_id"].astype(str)

    target_cache = args.target_cache or default_target_cache(args)
    print("Preparing waveform target cohort...")
    target_ids, target_matrix, target_metadata = build_or_load_target_cohort(
        features_df=features_df,
        manifest=manifest,
        records_dir=args.records_dir,
        cache_path=target_cache,
        target_records=args.target_records,
        cohort_seed=args.cohort_seed,
        segment_ref=args.segment_ref,
        target_fs=args.target_fs,
        workers=args.target_workers,
    )
    if target_matrix.shape[1] % 12 != 0:
        raise ValueError("Waveform target width is not divisible by 12 leads.")

    metrics_rows = []
    per_record_frames = []
    pca_rows = []
    encoder_protocols = []
    first_seed_examples = {}

    for seed in args.seeds:
        print(f"Seed {seed}: creating record-disjoint split and fitting target PCA...")
        train_df, test_df = split_segments_by_patient_stratified(
            features_df,
            group_col="patient_id",
            label_col="utility_label",
            test_size=args.test_size,
            random_state=seed,
        )
        train_cohort_ids = np.asarray(
            sorted(set(train_df["patient_id"].astype(str)).intersection(set(target_ids))), dtype=str
        )
        test_cohort_ids = np.asarray(
            sorted(set(test_df["patient_id"].astype(str)).intersection(set(target_ids))), dtype=str
        )
        y_train = align_targets(train_cohort_ids, target_ids, target_matrix)
        y_test = align_targets(test_cohort_ids, target_ids, target_matrix)

        n_components = min(
            args.target_pca_components,
            y_train.shape[0] - 1,
            y_train.shape[1],
        )
        target_pca = PCA(
            n_components=n_components,
            svd_solver="randomized",
            iterated_power=3,
            random_state=seed,
        )
        target_components_train = target_pca.fit_transform(y_train)
        target_scaler = StandardScaler().fit(target_components_train)
        pca_rows.append(
            {
                "seed": seed,
                "n_components": int(n_components),
                "explained_variance_ratio_sum": float(target_pca.explained_variance_ratio_.sum()),
                "train_targets": int(len(y_train)),
                "test_targets": int(len(y_test)),
            }
        )

        mean_prediction = np.repeat(y_train.mean(axis=0, keepdims=True), len(y_test), axis=0)
        baseline_metrics, baseline_records = reconstruction_metrics(
            patient_ids=test_cohort_ids,
            y_true_flat=y_test,
            y_pred_flat=mean_prediction,
            n_leads=12,
            target_fs=args.target_fs,
            max_lag_ms=args.max_lag_ms,
        )
        metrics_rows.append(
            {
                "seed": seed,
                "representation": "waveform_baseline",
                "decoder": "train_mean",
                "n_train": int(len(y_train)),
                "n_features": 0,
                "decoder_n_iter": np.nan,
                **baseline_metrics,
            }
        )
        baseline_records.insert(0, "decoder", "train_mean")
        baseline_records.insert(0, "representation", "waveform_baseline")
        baseline_records.insert(0, "seed", seed)
        per_record_frames.append(baseline_records)

        pca_ceiling_prediction = target_pca.inverse_transform(target_pca.transform(y_test))
        ceiling_metrics, ceiling_records = reconstruction_metrics(
            patient_ids=test_cohort_ids,
            y_true_flat=y_test,
            y_pred_flat=pca_ceiling_prediction,
            n_leads=12,
            target_fs=args.target_fs,
            max_lag_ms=args.max_lag_ms,
        )
        metrics_rows.append(
            {
                "seed": seed,
                "representation": "waveform_baseline",
                "decoder": "target_pca_ceiling",
                "n_train": int(len(y_train)),
                "n_features": int(n_components),
                "decoder_n_iter": np.nan,
                **ceiling_metrics,
            }
        )
        ceiling_records.insert(0, "decoder", "target_pca_ceiling")
        ceiling_records.insert(0, "representation", "waveform_baseline")
        ceiling_records.insert(0, "seed", seed)
        per_record_frames.append(ceiling_records)

        representation_splits = {"raw_features": (train_df, test_df)}
        print(f"Seed {seed}: fitting {args.bottleneck_dim}-D utility encoder...")
        feature_columns = get_feature_columns(features_df)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_encoder_df, test_encoder_df, encoder_metadata = build_supervised_encoder_representation(
                train_df=train_df,
                test_df=test_df,
                feature_columns=feature_columns,
                dim=args.bottleneck_dim,
                seed=seed,
                args=args,
            )
        encoder_protocols.append({"seed": seed, **encoder_metadata})
        representation_splits["encoder"] = (train_encoder_df, test_encoder_df)

        train_noisy_df, test_noisy_df, noise_metadata = apply_dp_to_embedding_split(
            train_embedding_df=train_encoder_df,
            test_embedding_df=test_encoder_df,
            epsilon=args.dp_epsilon,
            delta=args.dp_delta,
            clip_norm=args.dp_clip_norm,
            seed=seed + 30_000,
        )
        representation_splits["encoder_gaussian_noise"] = (train_noisy_df, test_noisy_df)

        for representation, (train_representation_df, test_representation_df) in representation_splits.items():
            print(f"Seed {seed}: training reconstruction decoders for {representation}...")
            rows, records, examples = evaluate_representation_decoder(
                representation=representation,
                train_df=train_representation_df,
                test_df=test_representation_df,
                target_ids=target_ids,
                target_matrix=target_matrix,
                target_pca=target_pca,
                target_scaler=target_scaler,
                seed=seed,
                args=args,
            )
            metrics_rows.extend(rows)
            per_record_frames.extend(records)
            if seed == args.seeds[0]:
                first_seed_examples.update(examples)

    metrics_df = pd.DataFrame(metrics_rows)
    per_record_df = pd.concat(per_record_frames, ignore_index=True)
    pca_df = pd.DataFrame(pca_rows)
    summary_df = summarize_metrics(metrics_df)
    per_lead_df = summarize_per_lead(per_record_df)

    metrics_path = args.output_dir / f"{args.run_name}_metrics.csv"
    summary_path = args.output_dir / f"{args.run_name}_summary.csv"
    per_record_path = args.output_dir / f"{args.run_name}_per_record.csv.gz"
    per_lead_path = args.output_dir / f"{args.run_name}_per_lead_summary.csv"
    pca_path = args.output_dir / f"{args.run_name}_target_pca.csv"
    protocol_path = args.output_dir / f"{args.run_name}_protocol.json"
    figure_path = args.figure_dir / f"{args.run_name}_lead_ii_examples.png"
    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    per_record_df.to_csv(per_record_path, index=False, compression="gzip")
    per_lead_df.to_csv(per_lead_path, index=False)
    pca_df.to_csv(pca_path, index=False)
    plot_examples(first_seed_examples, target_fs=args.target_fs, output_path=figure_path)

    protocol = {
        "run_name": args.run_name,
        "created_at_epoch": started,
        "elapsed_seconds": round(time.time() - started, 2),
        "dataset_dir": str(args.dataset_dir),
        "records_dir": str(args.records_dir),
        "loaded_chunks": len(chunk_files),
        "max_chunks": max_chunks,
        "loaded_segments": int(len(features_df)),
        "loaded_identifiers": int(features_df["patient_id"].nunique()),
        "seeds": args.seeds,
        "test_size": args.test_size,
        "split_unit": "retained record identifier",
        "target_cohort": target_metadata,
        "waveform_target": {
            "segment_ref": args.segment_ref,
            "preprocessing": "0.5-40 Hz bandpass followed by per-record, per-lead z-score normalization",
            "duration_seconds": manifest.get("window_sec", 2.0),
            "source_sampling_rate_hz": 500,
            "target_sampling_rate_hz": args.target_fs,
            "leads": 12,
            "samples_per_lead": int(target_matrix.shape[1] // 12),
            "target_pca_components": args.target_pca_components,
            "target_pca_note": (
                "The decoder predicts PCA coefficients fitted only on training waveforms; the target-PCA ceiling "
                "quantifies loss caused by this computational approximation."
            ),
        },
        "representations": ["raw_features", "encoder", "encoder_gaussian_noise"],
        "raw_representation_note": "raw_features denotes the 208 handcrafted segment features, not ECG samples",
        "decoders": {
            "models": args.decoders,
            "ridge_alpha": args.ridge_alpha,
            "mlp_hidden_units": args.decoder_hidden_units,
            "mlp_max_iter": args.decoder_max_iter,
            "mlp_batch_size": args.decoder_batch_size,
        },
        "encoder": {
            "bottleneck_dim": args.bottleneck_dim,
            "hidden_units": args.encoder_hidden_units,
            "max_iter": args.encoder_max_iter,
            "train_max_segments": args.encoder_train_max_segments,
            "per_seed": encoder_protocols,
        },
        "gaussian_noise": {
            **noise_metadata,
            "scope_note": (
                "The experiment reuses clipped standardized embeddings plus Gaussian noise. "
                "It is not evidence of end-to-end differential privacy."
            ),
        },
        "metrics": {
            "waveform_error": ["RMSE", "PRD", "SNR"],
            "shape_similarity": [
                "mean per-lead Pearson correlation",
                "per-lead Pearson correlation mean, standard deviation, median, and interquartile range",
                "per-lead PRD mean, standard deviation, median, and interquartile range",
                "Lead-II Pearson correlation",
                f"Lead-II maximum-lag correlation within +/-{args.max_lag_ms:g} ms",
                "Lead-II log-magnitude spectral correlation",
            ],
        },
        "claim_boundary": (
            "Poor reconstruction by the tested Ridge and MLP decoders is empirical attack evidence, not a proof "
            "that reconstruction is impossible against all attackers."
        ),
        "outputs": {
            "metrics": str(metrics_path),
            "summary": str(summary_path),
            "per_record": str(per_record_path),
            "per_lead_summary": str(per_lead_path),
            "target_pca": str(pca_path),
            "example_figure": str(figure_path),
        },
    }
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    print("Saved metrics:", metrics_path)
    print("Saved summary:", summary_path)
    print("Saved protocol:", protocol_path)


if __name__ == "__main__":
    main()
