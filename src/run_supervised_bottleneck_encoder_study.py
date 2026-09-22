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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train supervised bottleneck encoders for utility and evaluate linkability on learned embeddings."
    )
    parser.add_argument("--dataset-dir", type=Path, default=FINAL_SEGMENT_FEATURES_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_TABLES_DIR)
    parser.add_argument("--run-name", default="supervised_bottleneck_encoder")
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=20,
        help="Number of final dataset chunks to load. Use 0 to load all chunks.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--bottleneck-dims", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--hidden-units", type=int, default=128)
    parser.add_argument("--max-iter", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate-init", type=float, default=0.001)
    parser.add_argument(
        "--encoder-train-max-segments",
        type=int,
        default=80000,
        help="Balanced segment sample used to train the encoder. Use 0 to train on all train segments.",
    )
    parser.add_argument(
        "--train-noise-stds",
        type=float,
        nargs="+",
        default=[0.0],
        help="Optional Gaussian noise levels applied in standardized feature space while fitting the encoder.",
    )
    parser.add_argument("--utility-model", default="LogisticRegression")
    parser.add_argument("--link-max-pairs", type=int, default=2000)
    parser.add_argument("--link-min-segment-gap", type=int, default=4)
    parser.add_argument("--link-max-positive-pairs-per-patient", type=int, default=3)
    parser.add_argument("--negative-strategy", choices=["random", "hard"], default="random")
    parser.add_argument("--hard-negative-pool-size", type=int, default=5)
    return parser.parse_args()


def load_feature_dataset(dataset_dir: Path, max_chunks: int | None) -> tuple[pd.DataFrame, dict, list[Path]]:
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunk_files = [dataset_dir / item["chunk_file"] for item in manifest["chunks"]]
    if max_chunks is not None and max_chunks > 0:
        chunk_files = chunk_files[:max_chunks]

    features_df = pd.concat(
        [pd.read_csv(chunk_file, compression="gzip", low_memory=False) for chunk_file in chunk_files],
        ignore_index=True,
    )
    return features_df, manifest, chunk_files


def sample_balanced_encoder_rows(
    train_df: pd.DataFrame,
    target_col: str,
    max_segments: int,
    random_state: int,
) -> np.ndarray:
    indices = train_df.index.to_numpy()
    if max_segments is None or max_segments <= 0 or max_segments >= len(indices):
        return indices

    rng = np.random.default_rng(random_state)
    labels = train_df[target_col].astype(str)
    classes = sorted(labels.unique())
    per_class = max(1, max_segments // max(len(classes), 1))

    sampled = []
    leftovers = []
    for label in classes:
        class_indices = train_df.index[labels == label].to_numpy()
        rng.shuffle(class_indices)
        take = min(per_class, len(class_indices))
        sampled.extend(class_indices[:take])
        leftovers.extend(class_indices[take:])

    remaining = max_segments - len(sampled)
    if remaining > 0 and leftovers:
        leftovers = np.asarray(leftovers)
        take = min(remaining, len(leftovers))
        sampled.extend(rng.choice(leftovers, size=take, replace=False).tolist())

    sampled = np.asarray(sampled, dtype=int)
    rng.shuffle(sampled)
    return sampled


def activation_function(values: np.ndarray, activation: str) -> np.ndarray:
    if activation == "relu":
        return np.maximum(values, 0.0)
    if activation == "tanh":
        return np.tanh(values)
    if activation == "logistic":
        return 1.0 / (1.0 + np.exp(-values))
    if activation == "identity":
        return values
    raise ValueError(f"Unsupported MLP activation: {activation}")


def extract_bottleneck_embedding(model: MLPClassifier, matrix: np.ndarray) -> np.ndarray:
    activations = matrix
    for weights, bias in zip(model.coefs_[:-1], model.intercepts_[:-1]):
        activations = activation_function(activations @ weights + bias, model.activation)
    return activations.astype(np.float32, copy=False)


def build_embedding_dataframe(source_df: pd.DataFrame, embedding: np.ndarray, prefix: str = "enc") -> pd.DataFrame:
    metadata_columns = [col for col in source_df.columns if col in METADATA_COLUMNS]
    metadata_df = source_df[metadata_columns].reset_index(drop=True)
    embedding_df = pd.DataFrame(
        embedding,
        columns=[f"{prefix}_{idx:03d}" for idx in range(embedding.shape[1])],
    )
    return pd.concat([metadata_df, embedding_df], axis=1)


def evaluate_encoder_direct(
    model: MLPClassifier,
    label_encoder: LabelEncoder,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:  # pragma: no cover - MLPClassifier exposes predict_proba
        y_score = y_pred

    return {
        "encoder_direct_f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "encoder_direct_balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "encoder_direct_roc_auc": float(roc_auc_score(y_test, y_score)) if len(np.unique(y_test)) > 1 else np.nan,
        "encoder_direct_pr_auc": float(average_precision_score(y_test, y_score))
        if len(np.unique(y_test)) > 1
        else np.nan,
        "encoder_classes": [str(label) for label in label_encoder.classes_],
    }


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "utility_f1",
        "utility_balanced_accuracy",
        "utility_roc_auc",
        "utility_pr_auc",
        "encoder_direct_f1",
        "encoder_direct_balanced_accuracy",
        "encoder_direct_roc_auc",
        "encoder_direct_pr_auc",
        "linkability_f1",
        "linkability_balanced_accuracy",
        "linkability_roc_auc",
        "linkability_pr_auc",
    ]
    summary = metrics_df.groupby(
        ["bottleneck_dim", "train_noise_std", "n_embedding_features"],
        as_index=False,
    )[metric_cols].agg(["mean", "std"])
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
    protocol_rows = []

    for seed in args.seeds:
        train_df, test_df = split_segments_by_patient_stratified(
            features_df,
            group_col="patient_id",
            label_col="utility_label",
            test_size=args.test_size,
            random_state=seed,
        )

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
        X_encoder_base = X_train_full[encoder_positions]
        y_encoder = y_train_full[encoder_positions]

        for noise_std in args.train_noise_stds:
            rng = np.random.default_rng(seed + int(noise_std * 1_000_000))
            X_encoder = X_encoder_base.copy()
            if noise_std > 0:
                X_encoder = X_encoder + rng.normal(loc=0.0, scale=float(noise_std), size=X_encoder.shape)

            for bottleneck_dim in args.bottleneck_dims:
                print(
                    f"Seed {seed}: fitting bottleneck_dim={bottleneck_dim}, "
                    f"train_noise_std={noise_std}..."
                )
                model = MLPClassifier(
                    hidden_layer_sizes=(args.hidden_units, int(bottleneck_dim)),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    batch_size=args.batch_size,
                    learning_rate_init=args.learning_rate_init,
                    max_iter=args.max_iter,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=8,
                    random_state=seed,
                    verbose=False,
                )

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    model.fit(X_encoder, y_encoder)

                train_embedding = extract_bottleneck_embedding(model, X_train_full)
                test_embedding = extract_bottleneck_embedding(model, X_test)
                train_embedding_df = build_embedding_dataframe(train_df, train_embedding, prefix="enc")
                test_embedding_df = build_embedding_dataframe(test_df, test_embedding, prefix="enc")

                utility_out = evaluate_utility_model_on_split(
                    train_df=train_embedding_df,
                    test_df=test_embedding_df,
                    model_name=args.utility_model,
                    target_col="utility_label",
                )
                utility_eval = utility_out["evaluation"]

                encoder_direct_eval = evaluate_encoder_direct(
                    model=model,
                    label_encoder=label_encoder,
                    X_test=X_test,
                    y_test=y_test,
                )

                linkability_out = run_linkability_baselines_on_split(
                    train_df=train_embedding_df,
                    test_df=test_embedding_df,
                    max_pairs=args.link_max_pairs,
                    representation="absdiff",
                    random_state=seed,
                    min_segment_gap=args.link_min_segment_gap,
                    max_positive_pairs_per_patient=args.link_max_positive_pairs_per_patient,
                    negative_strategy=args.negative_strategy,
                    hard_negative_pool_size=args.hard_negative_pool_size,
                    include_distance_baselines=False,
                )
                linkability_summary = linkability_out["summary_df"]
                xgb_row = linkability_summary[linkability_summary["model"] == "XGBoost"].iloc[0]

                metrics_rows.append(
                    {
                        "run_name": args.run_name,
                        "seed": seed,
                        "bottleneck_dim": int(bottleneck_dim),
                        "hidden_units": args.hidden_units,
                        "train_noise_std": float(noise_std),
                        "n_input_features": len(feature_columns),
                        "n_embedding_features": int(train_embedding.shape[1]),
                        "encoder_train_segments": int(len(X_encoder)),
                        "train_segments": len(train_df),
                        "test_segments": len(test_df),
                        "train_patients": train_df["patient_id"].nunique(),
                        "test_patients": test_df["patient_id"].nunique(),
                        "positive_label": positive_name,
                        "negative_label": negative_name,
                        "utility_model": utility_eval["model"],
                        "utility_f1": utility_eval["f1_score"],
                        "utility_balanced_accuracy": utility_eval["balanced_accuracy"],
                        "utility_roc_auc": utility_eval["roc_auc"],
                        "utility_pr_auc": utility_eval["pr_auc"],
                        **encoder_direct_eval,
                        "linkability_model": xgb_row["model"],
                        "linkability_f1": xgb_row["f1_score"],
                        "linkability_balanced_accuracy": xgb_row["balanced_accuracy"],
                        "linkability_roc_auc": xgb_row["roc_auc"],
                        "linkability_pr_auc": xgb_row["pr_auc"],
                        "link_train_pairs": len(linkability_out["train_pair_df"]),
                        "link_test_pairs": len(linkability_out["test_pair_df"]),
                        "link_train_positive_prevalence": float(
                            linkability_out["train_pair_df"]["pair_label"].mean()
                        ),
                        "link_test_positive_prevalence": float(
                            linkability_out["test_pair_df"]["pair_label"].mean()
                        ),
                        "encoder_loss": float(model.loss_),
                        "encoder_n_iter": int(model.n_iter_),
                    }
                )

                protocol_rows.append(
                    {
                        "seed": seed,
                        "bottleneck_dim": int(bottleneck_dim),
                        "train_noise_std": float(noise_std),
                        "hidden_layer_sizes": [args.hidden_units, int(bottleneck_dim)],
                        "activation": model.activation,
                        "solver": model.solver,
                        "alpha": model.alpha,
                        "batch_size": args.batch_size,
                        "learning_rate_init": args.learning_rate_init,
                        "max_iter": args.max_iter,
                        "early_stopping": True,
                        "validation_fraction": 0.1,
                        "n_iter_no_change": 8,
                        "encoder_train_segments": int(len(X_encoder)),
                        "encoder_loss": float(model.loss_),
                        "encoder_n_iter": int(model.n_iter_),
                    }
                )

    metrics_df = pd.DataFrame(metrics_rows)
    summary_df = summarize_metrics(metrics_df)
    protocol_df = pd.DataFrame(protocol_rows)

    metrics_path = args.output_dir / f"{args.run_name}_metrics.csv"
    summary_path = args.output_dir / f"{args.run_name}_summary.csv"
    encoder_protocol_path = args.output_dir / f"{args.run_name}_encoder_protocol.csv"
    protocol_path = args.output_dir / f"{args.run_name}_protocol.json"

    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    protocol_df.to_csv(encoder_protocol_path, index=False)

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
        "test_size": args.test_size,
        "seeds": args.seeds,
        "bottleneck_dims": args.bottleneck_dims,
        "train_noise_stds": args.train_noise_stds,
        "encoder_protocol": {
            "model": "sklearn.neural_network.MLPClassifier",
            "objective": "supervised binary utility classification",
            "input_preprocessing": "median imputation and standard scaling fitted on train split",
            "training_rows": "balanced segment sample from train split only",
            "embedding": "activation of final hidden bottleneck layer",
            "hidden_units": args.hidden_units,
            "max_iter": args.max_iter,
            "batch_size": args.batch_size,
            "learning_rate_init": args.learning_rate_init,
            "encoder_train_max_segments": args.encoder_train_max_segments,
        },
        "utility_protocol": {
            "split": "patient-level stratified split before encoder fitting",
            "target": "utility_label",
            "model_on_embedding": args.utility_model,
        },
        "linkability_protocol": {
            "representation": "absdiff on supervised bottleneck embeddings",
            "model": "XGBoost",
            "max_pairs": args.link_max_pairs,
            "min_segment_gap": args.link_min_segment_gap,
            "max_positive_pairs_per_patient": args.link_max_positive_pairs_per_patient,
            "negative_strategy": args.negative_strategy,
            "hard_negative_pool_size": args.hard_negative_pool_size,
            "pair_prevalence": "balanced by construction",
        },
        "outputs": {
            "metrics": str(metrics_path),
            "summary": str(summary_path),
            "encoder_protocol": str(encoder_protocol_path),
        },
    }
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    print("Saved metrics:", metrics_path)
    print("Saved summary:", summary_path)
    print("Saved encoder protocol:", encoder_protocol_path)
    print("Saved protocol:", protocol_path)


if __name__ == "__main__":
    main()
