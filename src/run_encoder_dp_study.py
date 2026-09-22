from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from config import FINAL_SEGMENT_FEATURES_DIR, OUTPUTS_TABLES_DIR
    from modeling import (
        METADATA_COLUMNS,
        evaluate_utility_model_on_split,
        get_binary_targets,
        get_feature_columns,
        run_linkability_baselines_on_split,
        split_segments_by_patient_stratified,
    )
    from run_supervised_bottleneck_encoder_study import (
        build_embedding_dataframe,
        evaluate_encoder_direct,
        extract_bottleneck_embedding,
        load_feature_dataset,
        sample_balanced_encoder_rows,
    )
except ImportError:  # pragma: no cover - package import fallback
    from .config import FINAL_SEGMENT_FEATURES_DIR, OUTPUTS_TABLES_DIR
    from .modeling import (
        METADATA_COLUMNS,
        evaluate_utility_model_on_split,
        get_binary_targets,
        get_feature_columns,
        run_linkability_baselines_on_split,
        split_segments_by_patient_stratified,
    )
    from .run_supervised_bottleneck_encoder_study import (
        build_embedding_dataframe,
        evaluate_encoder_direct,
        extract_bottleneck_embedding,
        load_feature_dataset,
        sample_balanced_encoder_rows,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply DP-calibrated Gaussian perturbation after a supervised bottleneck encoder."
    )
    parser.add_argument("--dataset-dir", type=Path, default=FINAL_SEGMENT_FEATURES_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_TABLES_DIR)
    parser.add_argument("--run-name", default="encoder_dp_study")
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=20,
        help="Number of final dataset chunks to load. Use 0 to load all chunks.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--bottleneck-dim", type=int, default=8)
    parser.add_argument("--encoder-hidden-units", type=int, default=128)
    parser.add_argument("--encoder-max-iter", type=int, default=40)
    parser.add_argument("--encoder-batch-size", type=int, default=1024)
    parser.add_argument("--encoder-learning-rate-init", type=float, default=0.001)
    parser.add_argument("--encoder-train-max-segments", type=int, default=80000)
    parser.add_argument("--dp-epsilons", type=float, nargs="+", default=[100.0, 50.0, 20.0, 10.0])
    parser.add_argument("--dp-clip-norms", type=float, nargs="+", default=[2.0, 4.0])
    parser.add_argument("--dp-delta", type=float, default=1e-5)
    parser.add_argument("--utility-model", default="LogisticRegression")
    parser.add_argument("--link-max-pairs", type=int, default=2000)
    parser.add_argument("--link-min-segment-gap", type=int, default=4)
    parser.add_argument("--link-max-positive-pairs-per-patient", type=int, default=3)
    parser.add_argument("--negative-strategy", choices=["random", "hard"], default="random")
    parser.add_argument("--hard-negative-pool-size", type=int, default=5)
    return parser.parse_args()


def clip_rows_by_l2_norm(matrix: np.ndarray, clip_norm: float) -> np.ndarray:
    norms = np.linalg.norm(matrix, ord=2, axis=1, keepdims=True)
    scale = np.minimum(1.0, clip_norm / np.maximum(norms, 1e-12))
    return matrix * scale


def gaussian_dp_sigma(epsilon: float, delta: float, clip_norm: float) -> float:
    sensitivity = 2.0 * clip_norm
    return float(sensitivity * np.sqrt(2.0 * np.log(1.25 / delta)) / epsilon)


def copy_metadata_with_matrix(source_df: pd.DataFrame, matrix: np.ndarray, prefix: str) -> pd.DataFrame:
    metadata_columns = [col for col in source_df.columns if col in METADATA_COLUMNS]
    metadata_df = source_df[metadata_columns].reset_index(drop=True)
    feature_df = pd.DataFrame(matrix, columns=[f"{prefix}_{idx:03d}" for idx in range(matrix.shape[1])])
    return pd.concat([metadata_df, feature_df], axis=1)


def apply_dp_to_embedding_split(
    train_embedding_df: pd.DataFrame,
    test_embedding_df: pd.DataFrame,
    epsilon: float,
    delta: float,
    clip_norm: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    feature_columns = get_feature_columns(train_embedding_df)
    scaler = StandardScaler()
    train_matrix = scaler.fit_transform(train_embedding_df[feature_columns])
    test_matrix = scaler.transform(test_embedding_df[feature_columns])
    train_matrix = clip_rows_by_l2_norm(train_matrix, clip_norm=clip_norm)
    test_matrix = clip_rows_by_l2_norm(test_matrix, clip_norm=clip_norm)

    sigma = gaussian_dp_sigma(epsilon=epsilon, delta=delta, clip_norm=clip_norm)
    rng_train = np.random.default_rng(seed)
    rng_test = np.random.default_rng(seed + 10_000)
    train_noisy = train_matrix + rng_train.normal(loc=0.0, scale=sigma, size=train_matrix.shape)
    test_noisy = test_matrix + rng_test.normal(loc=0.0, scale=sigma, size=test_matrix.shape)

    train_out = copy_metadata_with_matrix(train_embedding_df, train_noisy, prefix="dp_enc")
    test_out = copy_metadata_with_matrix(test_embedding_df, test_noisy, prefix="dp_enc")
    metadata = {
        "dp_epsilon": float(epsilon),
        "dp_delta": float(delta),
        "dp_clip_norm": float(clip_norm),
        "dp_sensitivity": float(2.0 * clip_norm),
        "dp_noise_sigma": sigma,
        "dp_mechanism": "gaussian_l2",
        "dp_scope_note": (
            "Gaussian noise is calibrated for releasing each clipped standardized embedding vector. "
            "This does not make encoder training itself differentially private."
        ),
    }
    return train_out, test_out, metadata


def train_encoder_representation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    seed: int,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["utility_label"])
    y_train_full, positive_name, negative_name = get_binary_targets(
        label_encoder,
        train_df["utility_label"],
        preferred_labels=["Atrial", "AF", "Positive", "1"],
    )
    y_test, _, _ = get_binary_targets(
        label_encoder,
        test_df["utility_label"],
        preferred_labels=["Atrial", "AF", "Positive", "1"],
    )

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_full = imputer.fit_transform(train_df[feature_columns])
    X_test = imputer.transform(test_df[feature_columns])
    X_train_full = scaler.fit_transform(X_train_full)
    X_test = scaler.transform(X_test)

    encoder_indices = sample_balanced_encoder_rows(
        train_df=train_df,
        target_col="utility_label",
        max_segments=args.encoder_train_max_segments,
        random_state=seed,
    )
    index_to_position = pd.Series(np.arange(len(train_df)), index=train_df.index)
    encoder_positions = index_to_position.loc[encoder_indices].to_numpy(dtype=int)

    model = MLPClassifier(
        hidden_layer_sizes=(args.encoder_hidden_units, int(args.bottleneck_dim)),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=args.encoder_batch_size,
        learning_rate_init=args.encoder_learning_rate_init,
        max_iter=args.encoder_max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=8,
        random_state=seed,
        verbose=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X_train_full[encoder_positions], y_train_full[encoder_positions])

    train_embedding = extract_bottleneck_embedding(model, X_train_full)
    test_embedding = extract_bottleneck_embedding(model, X_test)
    train_embedding_df = build_embedding_dataframe(train_df, train_embedding, prefix="enc")
    test_embedding_df = build_embedding_dataframe(test_df, test_embedding, prefix="enc")
    direct_eval = evaluate_encoder_direct(model, label_encoder, X_test, y_test)
    metadata = {
        "encoder_direct": direct_eval,
        "encoder_loss": float(model.loss_),
        "encoder_n_iter": int(model.n_iter_),
        "encoder_train_segments": int(len(encoder_positions)),
        "positive_label": positive_name,
        "negative_label": negative_name,
    }
    return train_embedding_df, test_embedding_df, metadata


def evaluate_representation(
    train_repr_df: pd.DataFrame,
    test_repr_df: pd.DataFrame,
    seed: int,
    transform_name: str,
    transform_metadata: dict,
    encoder_metadata: dict,
    args: argparse.Namespace,
) -> dict:
    utility_out = evaluate_utility_model_on_split(
        train_df=train_repr_df,
        test_df=test_repr_df,
        model_name=args.utility_model,
        target_col="utility_label",
    )
    utility_eval = utility_out["evaluation"]

    linkability_out = run_linkability_baselines_on_split(
        train_df=train_repr_df,
        test_df=test_repr_df,
        max_pairs=args.link_max_pairs,
        representation="absdiff",
        random_state=seed,
        min_segment_gap=args.link_min_segment_gap,
        max_positive_pairs_per_patient=args.link_max_positive_pairs_per_patient,
        negative_strategy=args.negative_strategy,
        hard_negative_pool_size=args.hard_negative_pool_size,
        include_distance_baselines=False,
    )
    xgb_row = linkability_out["summary_df"][linkability_out["summary_df"]["model"] == "XGBoost"].iloc[0]
    direct = encoder_metadata.get("encoder_direct", {})

    return {
        "seed": seed,
        "transform_name": transform_name,
        "bottleneck_dim": int(args.bottleneck_dim),
        "n_output_features": len(get_feature_columns(train_repr_df)),
        "utility_model": utility_eval["model"],
        "utility_f1": float(utility_eval["f1_score"]),
        "utility_balanced_accuracy": float(utility_eval["balanced_accuracy"]),
        "utility_roc_auc": float(utility_eval["roc_auc"]),
        "utility_pr_auc": float(utility_eval["pr_auc"]),
        "encoder_direct_f1": direct.get("encoder_direct_f1"),
        "encoder_direct_balanced_accuracy": direct.get("encoder_direct_balanced_accuracy"),
        "encoder_direct_roc_auc": direct.get("encoder_direct_roc_auc"),
        "encoder_direct_pr_auc": direct.get("encoder_direct_pr_auc"),
        "linkability_model": xgb_row["model"],
        "linkability_f1": float(xgb_row["f1_score"]),
        "linkability_balanced_accuracy": float(xgb_row["balanced_accuracy"]),
        "linkability_roc_auc": float(xgb_row["roc_auc"]),
        "linkability_pr_auc": float(xgb_row["pr_auc"]),
        "link_train_pairs": len(linkability_out["train_pair_df"]),
        "link_test_pairs": len(linkability_out["test_pair_df"]),
        "encoder_loss": encoder_metadata.get("encoder_loss"),
        "encoder_n_iter": encoder_metadata.get("encoder_n_iter"),
        "encoder_train_segments": encoder_metadata.get("encoder_train_segments"),
        **transform_metadata,
    }


def summarize(metrics_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "transform_name",
        "bottleneck_dim",
        "n_output_features",
        "dp_epsilon",
        "dp_delta",
        "dp_clip_norm",
        "dp_noise_sigma",
    ]
    for col in group_cols:
        if col not in metrics_df.columns:
            metrics_df[col] = np.nan
    metric_cols = [
        "utility_f1",
        "utility_balanced_accuracy",
        "utility_roc_auc",
        "utility_pr_auc",
        "linkability_f1",
        "linkability_balanced_accuracy",
        "linkability_roc_auc",
        "linkability_pr_auc",
    ]
    summary = metrics_df.groupby(group_cols, dropna=False, as_index=False)[metric_cols].agg(["mean", "std"])
    summary.columns = [
        "_".join([part for part in col if part]) if isinstance(col, tuple) else col for col in summary.columns
    ]
    return summary


def main() -> None:
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    max_chunks = None if args.max_chunks == 0 else args.max_chunks
    features_df, manifest, chunk_files = load_feature_dataset(args.dataset_dir, max_chunks=max_chunks)
    feature_columns = get_feature_columns(features_df)

    metrics_rows = []
    for seed in args.seeds:
        train_df, test_df = split_segments_by_patient_stratified(
            features_df,
            group_col="patient_id",
            label_col="utility_label",
            test_size=args.test_size,
            random_state=seed,
        )
        print(f"Seed {seed}: fitting encoder...")
        train_embedding_df, test_embedding_df, encoder_metadata = train_encoder_representation(
            train_df=train_df,
            test_df=test_df,
            feature_columns=feature_columns,
            seed=seed,
            args=args,
        )

        metrics_rows.append(
            evaluate_representation(
                train_repr_df=train_embedding_df,
                test_repr_df=test_embedding_df,
                seed=seed,
                transform_name="encoder_identity",
                transform_metadata={
                    "dp_epsilon": np.nan,
                    "dp_delta": np.nan,
                    "dp_clip_norm": np.nan,
                    "dp_noise_sigma": np.nan,
                    "dp_mechanism": "none",
                    "dp_scope_note": "No DP noise applied.",
                },
                encoder_metadata=encoder_metadata,
                args=args,
            )
        )

        for clip_norm in args.dp_clip_norms:
            for epsilon in args.dp_epsilons:
                transform_name = f"encoder_dp_gaussian_eps{epsilon:g}_clip{clip_norm:g}"
                print(f"Seed {seed}: evaluating {transform_name}...")
                train_dp_df, test_dp_df, dp_metadata = apply_dp_to_embedding_split(
                    train_embedding_df=train_embedding_df,
                    test_embedding_df=test_embedding_df,
                    epsilon=float(epsilon),
                    delta=float(args.dp_delta),
                    clip_norm=float(clip_norm),
                    seed=seed + int(float(epsilon) * 10) + int(float(clip_norm) * 1000),
                )
                metrics_rows.append(
                    evaluate_representation(
                        train_repr_df=train_dp_df,
                        test_repr_df=test_dp_df,
                        seed=seed,
                        transform_name=transform_name,
                        transform_metadata=dp_metadata,
                        encoder_metadata=encoder_metadata,
                        args=args,
                    )
                )

    metrics_df = pd.DataFrame(metrics_rows)
    summary_df = summarize(metrics_df)

    metrics_path = args.output_dir / f"{args.run_name}_metrics.csv"
    summary_path = args.output_dir / f"{args.run_name}_summary.csv"
    protocol_path = args.output_dir / f"{args.run_name}_protocol.json"
    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    protocol = {
        "run_name": args.run_name,
        "created_at_epoch": started,
        "elapsed_seconds": round(time.time() - started, 2),
        "dataset_dir": str(args.dataset_dir),
        "manifest_total_segments": manifest.get("total_segments"),
        "manifest_window_sec": manifest.get("window_sec"),
        "manifest_step_sec": manifest.get("step_sec"),
        "loaded_chunks": len(chunk_files),
        "max_chunks": max_chunks,
        "loaded_segments": len(features_df),
        "loaded_identifiers": int(features_df["patient_id"].nunique()),
        "seeds": args.seeds,
        "test_size": args.test_size,
        "bottleneck_dim": args.bottleneck_dim,
        "encoder_hidden_units": args.encoder_hidden_units,
        "encoder_max_iter": args.encoder_max_iter,
        "encoder_train_max_segments": args.encoder_train_max_segments,
        "dp_epsilons": args.dp_epsilons,
        "dp_clip_norms": args.dp_clip_norms,
        "dp_delta": args.dp_delta,
        "dp_protocol": {
            "mechanism": "Gaussian mechanism after L2 clipping in standardized embedding space",
            "sensitivity": "2 * clip_norm per released embedding vector",
            "sigma": "2 * clip_norm * sqrt(2 * log(1.25 / delta)) / epsilon",
            "scope_note": (
                "This is a DP-calibrated perturbation of the released embedding. "
                "It is not DP-SGD and does not make encoder training itself differentially private."
            ),
        },
        "utility_model": args.utility_model,
        "linkability_protocol": {
            "representation": "absdiff",
            "model": "XGBoost",
            "max_pairs": args.link_max_pairs,
            "min_segment_gap": args.link_min_segment_gap,
            "max_positive_pairs_per_patient": args.link_max_positive_pairs_per_patient,
            "negative_strategy": args.negative_strategy,
            "hard_negative_pool_size": args.hard_negative_pool_size,
        },
        "metrics_path": str(metrics_path),
        "summary_path": str(summary_path),
    }
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    print("Saved metrics:", metrics_path)
    print("Saved summary:", summary_path)
    print("Saved protocol:", protocol_path)


if __name__ == "__main__":
    main()
