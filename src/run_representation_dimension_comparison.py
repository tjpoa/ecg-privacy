from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from config import FEATURE_SETS_DIR, FINAL_SEGMENT_FEATURES_DIR, OUTPUTS_FIGURES_DIR, OUTPUTS_TABLES_DIR
    from modeling import (
        METADATA_COLUMNS,
        evaluate_utility_model_on_split,
        get_binary_targets,
        get_feature_columns,
        run_linkability_baselines_on_split,
        split_segments_by_patient_stratified,
    )
    from privacy_transforms import fit_transform_feature_split
    from run_encoder_dp_study import apply_dp_to_embedding_split
    from run_supervised_bottleneck_encoder_study import (
        build_embedding_dataframe,
        evaluate_encoder_direct,
        extract_bottleneck_embedding,
        sample_balanced_encoder_rows,
    )
except ImportError:  # pragma: no cover - package import fallback
    from .config import FEATURE_SETS_DIR, FINAL_SEGMENT_FEATURES_DIR, OUTPUTS_FIGURES_DIR, OUTPUTS_TABLES_DIR
    from .modeling import (
        METADATA_COLUMNS,
        evaluate_utility_model_on_split,
        get_binary_targets,
        get_feature_columns,
        run_linkability_baselines_on_split,
        split_segments_by_patient_stratified,
    )
    from .privacy_transforms import fit_transform_feature_split
    from .run_encoder_dp_study import apply_dp_to_embedding_split
    from .run_supervised_bottleneck_encoder_study import (
        build_embedding_dataframe,
        evaluate_encoder_direct,
        extract_bottleneck_embedding,
        sample_balanced_encoder_rows,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare representation dimension curves for supervised encoder, PCA, random projection, and top-k utility features."
    )
    parser.add_argument("--dataset-dir", type=Path, default=FINAL_SEGMENT_FEATURES_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_TABLES_DIR)
    parser.add_argument("--figures-dir", type=Path, default=OUTPUTS_FIGURES_DIR)
    parser.add_argument("--run-name", default="representation_dimension_comparison")
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=20,
        help="Number of final dataset chunks to load. Use 0 to load all chunks.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--dims", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["supervised_encoder", "pca", "random_projection", "topk_utility"],
        choices=[
            "identity",
            "supervised_encoder",
            "encoder_gaussian_c2",
            "encoder_gaussian_c4",
            "pca",
            "random_projection",
            "topk_utility",
        ],
    )
    parser.add_argument(
        "--method-dims",
        nargs="+",
        default=None,
        help=(
            "Optional method-specific dimensions, for example "
            "pca:64 random_projection:64 supervised_encoder:8."
        ),
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--utility-model", default="LogisticRegression")
    parser.add_argument("--link-max-pairs", type=int, default=2000)
    parser.add_argument("--link-min-segment-gap", type=int, default=4)
    parser.add_argument("--link-max-positive-pairs-per-patient", type=int, default=3)
    parser.add_argument("--negative-strategy", choices=["random", "hard"], default="random")
    parser.add_argument("--hard-negative-pool-size", type=int, default=5)
    parser.add_argument("--encoder-hidden-units", type=int, default=128)
    parser.add_argument("--encoder-max-iter", type=int, default=40)
    parser.add_argument("--encoder-batch-size", type=int, default=1024)
    parser.add_argument("--encoder-learning-rate-init", type=float, default=0.001)
    parser.add_argument("--encoder-train-max-segments", type=int, default=80000)
    parser.add_argument(
        "--shared-train-max-segments",
        type=int,
        default=0,
        help=(
            "If positive, select one label-balanced training subset per seed and use exactly that subset "
            "to fit and evaluate every representation."
        ),
    )
    parser.add_argument("--gaussian-epsilon", type=float, default=50.0)
    parser.add_argument("--gaussian-delta", type=float, default=1e-5)
    return parser.parse_args()


def parse_method_dims(values: list[str] | None) -> dict[str, list[int]]:
    if not values:
        return {}
    parsed: dict[str, list[int]] = {}
    for value in values:
        if ":" not in value:
            raise ValueError(f"Method dimension must use method:dimension format, got {value!r}.")
        method, raw_dim = value.split(":", 1)
        dim = int(raw_dim)
        if dim <= 0:
            raise ValueError(f"Representation dimension must be positive, got {value!r}.")
        parsed.setdefault(method, []).append(dim)
    return parsed


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


def copy_metadata_with_features(source_df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    metadata_columns = [col for col in source_df.columns if col in METADATA_COLUMNS]
    return pd.concat([source_df[metadata_columns].reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)


def load_utility_ranked_features(feature_columns: list[str]) -> list[str]:
    path = FEATURE_SETS_DIR / "utility_top150_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing utility feature ranking: {path}")
    ranking = pd.read_csv(path)
    feature_col = "feature" if "feature" in ranking.columns else ranking.columns[0]
    available = set(feature_columns)
    ranked = []
    for feature in ranking[feature_col].dropna().astype(str):
        if feature in available and feature not in ranked:
            ranked.append(feature)
    if not ranked:
        raise ValueError(f"No ranked utility features from {path} matched the current feature columns.")
    return ranked


def build_topk_representation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ranked_features: list[str],
    dim: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    selected = ranked_features[:dim]
    train_out = copy_metadata_with_features(train_df, train_df[selected].copy())
    test_out = copy_metadata_with_features(test_df, test_df[selected].copy())
    return train_out, test_out, {"selected_features": selected}


def build_supervised_encoder_representation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    dim: int,
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
    index_to_position = pd.Series(range(len(train_df)), index=train_df.index)
    encoder_positions = index_to_position.loc[encoder_indices].to_numpy(dtype=int)

    model = MLPClassifier(
        hidden_layer_sizes=(args.encoder_hidden_units, int(dim)),
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
    train_out = build_embedding_dataframe(train_df, train_embedding, prefix="enc")
    test_out = build_embedding_dataframe(test_df, test_embedding, prefix="enc")
    direct_eval = evaluate_encoder_direct(model, label_encoder, X_test, y_test)
    metadata = {
        "encoder_direct": direct_eval,
        "encoder_loss": float(model.loss_),
        "encoder_n_iter": int(model.n_iter_),
        "encoder_train_segments": int(len(encoder_positions)),
        "positive_label": positive_name,
        "negative_label": negative_name,
    }
    return train_out, test_out, metadata


def build_representation(
    method: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    ranked_features: list[str],
    dim: int,
    seed: int,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if method == "identity":
        return train_df.copy(), test_df.copy(), {}

    if method == "pca":
        train_out, test_out, fitted = fit_transform_feature_split(
            train_df=train_df,
            test_df=test_df,
            method="pca",
            feature_columns=feature_columns,
            n_components=dim,
            random_state=seed,
        )
        return train_out, test_out, {
            "explained_variance_ratio_sum": float(fitted.pca.explained_variance_ratio_.sum())
        }

    if method == "random_projection":
        train_out, test_out, _ = fit_transform_feature_split(
            train_df=train_df,
            test_df=test_df,
            method="random_projection",
            feature_columns=feature_columns,
            n_components=dim,
            random_state=seed,
        )
        return train_out, test_out, {}

    if method == "topk_utility":
        return build_topk_representation(train_df, test_df, ranked_features, dim)

    if method == "supervised_encoder":
        return build_supervised_encoder_representation(train_df, test_df, feature_columns, dim, seed, args)

    if method in {"encoder_gaussian_c2", "encoder_gaussian_c4"}:
        train_embedding, test_embedding, metadata = build_supervised_encoder_representation(
            train_df,
            test_df,
            feature_columns,
            dim,
            seed,
            args,
        )
        clip_norm = 2.0 if method == "encoder_gaussian_c2" else 4.0
        train_out, test_out, noise_metadata = apply_dp_to_embedding_split(
            train_embedding_df=train_embedding,
            test_embedding_df=test_embedding,
            epsilon=args.gaussian_epsilon,
            delta=args.gaussian_delta,
            clip_norm=clip_norm,
            seed=seed + int(args.gaussian_epsilon * 10) + int(clip_norm * 1000),
        )
        return train_out, test_out, {**metadata, **noise_metadata}

    raise ValueError(f"Unsupported method: {method}")


def evaluate_representation(
    method: str,
    dim: int,
    seed: int,
    train_repr_df: pd.DataFrame,
    test_repr_df: pd.DataFrame,
    representation_metadata: dict,
    args: argparse.Namespace,
) -> tuple[dict, pd.DataFrame]:
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
    linkability_row = linkability_out["summary_df"][linkability_out["summary_df"]["model"] == "XGBoost"].iloc[0]

    direct = representation_metadata.get("encoder_direct", {})
    row = {
        "seed": seed,
        "method": method,
        "dimension": int(dim),
        "n_output_features": len(get_feature_columns(train_repr_df)),
        "utility_model": utility_eval["model"],
        "utility_f1": utility_eval["f1_score"],
        "utility_balanced_accuracy": utility_eval["balanced_accuracy"],
        "utility_roc_auc": utility_eval["roc_auc"],
        "utility_pr_auc": utility_eval["pr_auc"],
        "encoder_direct_f1": direct.get("encoder_direct_f1"),
        "encoder_direct_balanced_accuracy": direct.get("encoder_direct_balanced_accuracy"),
        "encoder_direct_roc_auc": direct.get("encoder_direct_roc_auc"),
        "encoder_direct_pr_auc": direct.get("encoder_direct_pr_auc"),
        "linkability_model": linkability_row["model"],
        "linkability_f1": linkability_row["f1_score"],
        "linkability_balanced_accuracy": linkability_row["balanced_accuracy"],
        "linkability_roc_auc": linkability_row["roc_auc"],
        "linkability_pr_auc": linkability_row["pr_auc"],
        "link_train_pairs": len(linkability_out["train_pair_df"]),
        "link_test_pairs": len(linkability_out["test_pair_df"]),
        "explained_variance_ratio_sum": representation_metadata.get("explained_variance_ratio_sum"),
        "encoder_loss": representation_metadata.get("encoder_loss"),
        "encoder_n_iter": representation_metadata.get("encoder_n_iter"),
        "encoder_train_segments": representation_metadata.get("encoder_train_segments"),
        "noise_multiplier": (
            representation_metadata.get("dp_noise_sigma") / representation_metadata.get("dp_clip_norm")
            if representation_metadata.get("dp_clip_norm")
            else np.nan
        ),
        "clip_norm": representation_metadata.get("dp_clip_norm"),
    }
    utility_predictions = utility_out["predictions_df"].copy()
    utility_predictions["seed"] = seed
    utility_predictions["method"] = method
    utility_predictions["dimension"] = int(dim)
    return row, utility_predictions


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
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
    summary = metrics_df.groupby(["method", "dimension", "n_output_features"], as_index=False)[metric_cols].agg(
        ["mean", "std"]
    )
    summary.columns = [
        "_".join([part for part in col if part]) if isinstance(col, tuple) else col for col in summary.columns
    ]
    counts = (
        metrics_df.groupby(["method", "dimension", "n_output_features"], as_index=False)
        .agg(n_seeds=("seed", "nunique"), n_evaluations=("seed", "count"))
    )
    summary = summary.merge(counts, on=["method", "dimension", "n_output_features"], how="left")
    return summary


def save_outputs(metrics_rows: list[dict], args: argparse.Namespace) -> tuple[Path, Path]:
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = args.output_dir / f"{args.run_name}_metrics.csv"
    summary_path = args.output_dir / f"{args.run_name}_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)
    if not metrics_df.empty:
        summarize_metrics(metrics_df).to_csv(summary_path, index=False)
    return metrics_path, summary_path


def save_figure(summary_path: Path, args: argparse.Namespace) -> Path | None:
    if not summary_path.exists():
        return None
    summary = pd.read_csv(summary_path)
    if summary.empty:
        return None
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(8.5, 5))
    ax2 = ax1.twinx()
    for method, group in summary.groupby("method"):
        group = group.sort_values("dimension")
        ax1.plot(group["dimension"], group["utility_f1_mean"], marker="o", label=f"{method} utility F1")
        ax2.plot(
            group["dimension"],
            group["linkability_roc_auc_mean"],
            marker="s",
            linestyle="--",
            label=f"{method} link ROC-AUC",
        )
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Representation dimension")
    ax1.set_ylabel("Utility F1")
    ax2.set_ylabel("Linkability ROC-AUC")
    ax1.set_title("Representation dimension vs utility/linkability")
    ax1.grid(alpha=0.25)
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=7, loc="center right")
    fig.tight_layout()
    fig_path = args.figures_dir / f"{args.run_name}.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def main() -> None:
    args = parse_args()
    method_dims = parse_method_dims(args.method_dims)
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    max_chunks = None if args.max_chunks == 0 else args.max_chunks
    features_df, manifest, chunk_files = load_feature_dataset(args.dataset_dir, max_chunks=max_chunks)
    feature_columns = get_feature_columns(features_df)
    ranked_features = load_utility_ranked_features(feature_columns)

    metrics_rows = []
    utility_prediction_frames: list[pd.DataFrame] = []
    split_rows: list[dict[str, object]] = []
    metrics_path = args.output_dir / f"{args.run_name}_metrics.csv"
    summary_path = args.output_dir / f"{args.run_name}_summary.csv"
    predictions_path = args.output_dir / f"{args.run_name}_utility_predictions.csv"
    splits_path = args.output_dir / f"{args.run_name}_splits.csv"
    protocol_path = args.output_dir / f"{args.run_name}_protocol.json"

    for seed in args.seeds:
        train_df, test_df = split_segments_by_patient_stratified(
            features_df,
            group_col="patient_id",
            label_col="utility_label",
            test_size=args.test_size,
            random_state=seed,
        )
        full_train_df = train_df
        if args.shared_train_max_segments and args.shared_train_max_segments > 0:
            shared_indices = sample_balanced_encoder_rows(
                train_df=full_train_df,
                target_col="utility_label",
                max_segments=args.shared_train_max_segments,
                random_state=seed,
            )
            train_df = full_train_df.loc[shared_indices].reset_index(drop=True)
        train_ids = sorted(str(value) for value in train_df["patient_id"].unique())
        test_ids = sorted(str(value) for value in test_df["patient_id"].unique())
        split_rows.append({
            "seed": seed,
            "group_column": "patient_id (interpreted as retained_record_id)",
            "n_train_rows": len(train_df),
            "n_full_train_rows": len(full_train_df),
            "n_test_rows": len(test_df),
            "n_train_groups": len(train_ids),
            "n_full_train_groups": int(full_train_df["patient_id"].nunique()),
            "n_test_groups": len(test_ids),
            "train_groups": "|".join(train_ids),
            "test_groups": "|".join(test_ids),
            "group_overlap": 0,
        })

        methods = list(args.methods)
        if "identity" in methods:
            methods = ["identity"] + [method for method in methods if method != "identity"]

        for method in methods:
            if method in method_dims:
                dims = method_dims[method]
            else:
                dims = [len(feature_columns)] if method == "identity" else args.dims
            for dim in dims:
                print(f"Seed {seed}: method={method}, dimension={dim}...")
                train_repr_df, test_repr_df, metadata = build_representation(
                    method=method,
                    train_df=train_df,
                    test_df=test_df,
                    feature_columns=feature_columns,
                    ranked_features=ranked_features,
                    dim=int(dim),
                    seed=seed,
                    args=args,
                )
                row, utility_predictions = evaluate_representation(
                    method=method,
                    dim=int(dim),
                    seed=seed,
                    train_repr_df=train_repr_df,
                    test_repr_df=test_repr_df,
                    representation_metadata=metadata,
                    args=args,
                )
                row["run_name"] = args.run_name
                metrics_rows.append(row)
                utility_predictions["run_name"] = args.run_name
                utility_prediction_frames.append(utility_predictions)
                save_outputs(metrics_rows, args)

    metrics_path, summary_path = save_outputs(metrics_rows, args)
    utility_predictions_df = (
        pd.concat(utility_prediction_frames, ignore_index=True)
        if utility_prediction_frames
        else pd.DataFrame()
    )
    utility_predictions_df.to_csv(predictions_path, index=False)
    pd.DataFrame(split_rows).to_csv(splits_path, index=False)
    fig_path = save_figure(summary_path, args)

    protocol = {
        "run_name": args.run_name,
        "created_at_epoch": started,
        "elapsed_seconds": round(time.time() - started, 2),
        "dataset_dir": str(args.dataset_dir),
        "loaded_chunks": len(chunk_files),
        "max_chunks": max_chunks,
        "loaded_segments": len(features_df),
        "loaded_identifiers": int(features_df["patient_id"].nunique()),
        "manifest_total_segments": manifest.get("total_segments"),
        "manifest_window_sec": manifest.get("window_sec"),
        "manifest_step_sec": manifest.get("step_sec"),
        "seeds": args.seeds,
        "dims": args.dims,
        "methods": args.methods,
        "test_size": args.test_size,
        "utility_protocol": {
            "model": args.utility_model,
            "target": "utility_label",
            "split": "patient-level stratified split before representation fitting",
            "shared_train_max_segments": args.shared_train_max_segments,
            "shared_train_note": (
                "When positive, the same label-balanced training rows are used for every representation."
            ),
        },
        "linkability_protocol": {
            "representation": "absdiff on representation features",
            "model": "XGBoost",
            "max_pairs": args.link_max_pairs,
            "min_segment_gap": args.link_min_segment_gap,
            "max_positive_pairs_per_patient": args.link_max_positive_pairs_per_patient,
            "negative_strategy": args.negative_strategy,
        },
        "encoder_protocol": {
            "model": "sklearn.neural_network.MLPClassifier",
            "hidden_units": args.encoder_hidden_units,
            "max_iter": args.encoder_max_iter,
            "encoder_train_max_segments": args.encoder_train_max_segments,
            "embedding": "final hidden bottleneck activation",
        },
        "gaussian_perturbation": {
            "epsilon": args.gaussian_epsilon,
            "delta": args.gaussian_delta,
            "clip_norms": [2.0, 4.0],
            "scope": "Post-processing of the utility encoder; not an end-to-end DP guarantee.",
        },
        "outputs": {
            "metrics": str(metrics_path),
            "summary": str(summary_path),
            "utility_predictions": str(predictions_path),
            "group_splits": str(splits_path),
            "figure": str(fig_path) if fig_path is not None else None,
        },
        "leakage_checks": {
            "group_disjoint_assertion": True,
            "group_column": "patient_id (interpreted analytically as retained_record_id)",
            "representation_fitting_scope": "training split only",
        },
    }
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print("Saved metrics:", metrics_path)
    print("Saved summary:", summary_path)
    print("Saved utility predictions:", predictions_path)
    print("Saved group splits:", splits_path)
    print("Saved figure:", fig_path)
    print("Saved protocol:", protocol_path)


if __name__ == "__main__":
    main()
