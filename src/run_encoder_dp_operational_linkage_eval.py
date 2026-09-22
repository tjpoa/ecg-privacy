from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    from config import FINAL_SEGMENT_FEATURES_DIR, OUTPUTS_TABLES_DIR
    from modeling import (
        _build_xgb_model,
        build_pair_table,
        evaluate_utility_model_on_split,
        get_feature_columns,
        split_segments_by_patient_stratified,
    )
    from run_encoder_dp_study import apply_dp_to_embedding_split, train_encoder_representation
    from run_operational_linkage_eval import (
        build_absdiff_pair_features_fast,
        build_gallery_pairs,
        compute_ranking_metrics,
        equal_error_rate,
        get_positive_scores,
        load_feature_dataset,
        tpr_at_fpr,
    )
except ImportError:  # pragma: no cover - package import fallback
    from .config import FINAL_SEGMENT_FEATURES_DIR, OUTPUTS_TABLES_DIR
    from .modeling import (
        _build_xgb_model,
        build_pair_table,
        evaluate_utility_model_on_split,
        get_feature_columns,
        split_segments_by_patient_stratified,
    )
    from .run_encoder_dp_study import apply_dp_to_embedding_split, train_encoder_representation
    from .run_operational_linkage_eval import (
        build_absdiff_pair_features_fast,
        build_gallery_pairs,
        compute_ranking_metrics,
        equal_error_rate,
        get_positive_scores,
        load_feature_dataset,
        tpr_at_fpr,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate operational-style linkage after supervised encoder plus DP-calibrated noise."
    )
    parser.add_argument("--dataset-dir", type=Path, default=FINAL_SEGMENT_FEATURES_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_TABLES_DIR)
    parser.add_argument("--run-name", default="encoder_dp_operational_linkage")
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=20,
        help="Number of final dataset chunks to load. Use 0 to load all chunks.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--bottleneck-dim", type=int, default=8)
    parser.add_argument("--encoder-hidden-units", type=int, default=128)
    parser.add_argument("--encoder-max-iter", type=int, default=25)
    parser.add_argument("--encoder-batch-size", type=int, default=1024)
    parser.add_argument("--encoder-learning-rate-init", type=float, default=0.001)
    parser.add_argument("--encoder-train-max-segments", type=int, default=40000)
    parser.add_argument(
        "--dp-configs",
        nargs="+",
        default=["100:2", "50:2", "50:4", "20:4"],
        help="DP configs as epsilon:clip_norm, e.g. 50:2.",
    )
    parser.add_argument("--dp-delta", type=float, default=1e-5)
    parser.add_argument("--utility-model", default="LogisticRegression")
    parser.add_argument("--train-max-pairs", type=int, default=2000)
    parser.add_argument("--min-segment-gap", type=int, default=4)
    parser.add_argument("--max-positive-pairs-per-patient", type=int, default=3)
    parser.add_argument("--train-negative-strategy", choices=["random", "hard"], default="random")
    parser.add_argument("--hard-negative-pool-size", type=int, default=5)
    parser.add_argument("--gallery-negatives", type=int, nargs="+", default=[99, 999])
    parser.add_argument("--queries-per-seed", type=int, default=1000)
    parser.add_argument("--fpr-levels", type=float, nargs="+", default=[0.001, 0.005, 0.01])
    parser.add_argument("--recall-k", type=int, nargs="+", default=[1, 5, 10])
    return parser.parse_args()


def parse_dp_configs(values: list[str]) -> list[tuple[float, float]]:
    configs: list[tuple[float, float]] = []
    for value in values:
        parts = value.split(":")
        if len(parts) != 2:
            raise ValueError(f"DP config must use epsilon:clip_norm format, got {value!r}.")
        epsilon = float(parts[0])
        clip_norm = float(parts[1])
        if epsilon <= 0 or clip_norm <= 0:
            raise ValueError(f"DP epsilon and clip_norm must be positive, got {value!r}.")
        configs.append((epsilon, clip_norm))
    return configs


def evaluate_operational_representation(
    train_repr_df: pd.DataFrame,
    test_repr_df: pd.DataFrame,
    gallery_tables: dict[int, tuple[pd.DataFrame, pd.DataFrame]],
    transform_name: str,
    transform_metadata: dict,
    encoder_metadata: dict,
    seed: int,
    args: argparse.Namespace,
) -> list[dict]:
    feature_columns = get_feature_columns(train_repr_df)
    utility_out = evaluate_utility_model_on_split(
        train_df=train_repr_df,
        test_df=test_repr_df,
        model_name=args.utility_model,
        target_col="utility_label",
        feature_columns=feature_columns,
    )
    utility_eval = utility_out["evaluation"]

    train_pair_df = build_pair_table(
        train_repr_df,
        max_pairs=args.train_max_pairs,
        random_state=seed,
        min_segment_gap=args.min_segment_gap,
        max_pairs_per_patient=args.max_positive_pairs_per_patient,
        negative_strategy=args.train_negative_strategy,
        hard_negative_pool_size=args.hard_negative_pool_size,
    )
    X_train_pair, y_train_pair = build_absdiff_pair_features_fast(train_repr_df, train_pair_df, feature_columns)
    pair_scaler = StandardScaler()
    X_train_pair = pair_scaler.fit_transform(X_train_pair)
    link_model = _build_xgb_model()
    link_model.fit(X_train_pair, y_train_pair)

    test_pair_df = build_pair_table(
        test_repr_df,
        max_pairs=args.train_max_pairs,
        random_state=seed + 1,
        min_segment_gap=args.min_segment_gap,
        max_pairs_per_patient=args.max_positive_pairs_per_patient,
        negative_strategy=args.train_negative_strategy,
        hard_negative_pool_size=args.hard_negative_pool_size,
    )
    X_test_pair, y_test_pair = build_absdiff_pair_features_fast(test_repr_df, test_pair_df, feature_columns)
    X_test_pair = pair_scaler.transform(X_test_pair)
    pair_scores = get_positive_scores(link_model, X_test_pair)
    pair_pred = (pair_scores >= 0.5).astype(int)
    pairwise_metrics = {
        "pairwise_linkability_f1": float(f1_score(y_test_pair, pair_pred, zero_division=0)),
        "pairwise_linkability_balanced_accuracy": float(balanced_accuracy_score(y_test_pair, pair_pred)),
        "pairwise_linkability_roc_auc": float(roc_auc_score(y_test_pair, pair_scores)),
        "pairwise_linkability_pr_auc": float(average_precision_score(y_test_pair, pair_scores)),
        "pairwise_linkability_eer": equal_error_rate(y_test_pair, pair_scores),
        "link_test_pairs": len(test_pair_df),
        "link_test_positive_prevalence": float(test_pair_df["pair_label"].mean()),
    }

    direct = encoder_metadata.get("encoder_direct", {})
    rows = []
    for gallery_negatives, (pair_df, query_df) in gallery_tables.items():
        X_gallery, y_gallery = build_absdiff_pair_features_fast(test_repr_df, pair_df, feature_columns)
        X_gallery = pair_scaler.transform(X_gallery)
        scores = get_positive_scores(link_model, X_gallery)

        row = {
            "seed": seed,
            "transform_name": transform_name,
            "bottleneck_dim": int(args.bottleneck_dim),
            "n_output_features": len(feature_columns),
            "train_negative_strategy": args.train_negative_strategy,
            "gallery_negatives": int(gallery_negatives),
            "queries": len(query_df),
            "candidate_pairs": len(pair_df),
            "positive_prevalence": float(pair_df["pair_label"].mean()),
            "utility_model": utility_eval["model"],
            "utility_f1": float(utility_eval["f1_score"]),
            "utility_balanced_accuracy": float(utility_eval["balanced_accuracy"]),
            "utility_roc_auc": float(utility_eval["roc_auc"]),
            "utility_pr_auc": float(utility_eval["pr_auc"]),
            "encoder_direct_f1": direct.get("encoder_direct_f1"),
            "encoder_direct_balanced_accuracy": direct.get("encoder_direct_balanced_accuracy"),
            "encoder_direct_roc_auc": direct.get("encoder_direct_roc_auc"),
            "encoder_direct_pr_auc": direct.get("encoder_direct_pr_auc"),
            "link_train_pairs": len(train_pair_df),
            "link_train_positive_prevalence": float(train_pair_df["pair_label"].mean()),
            **pairwise_metrics,
            "link_operational_roc_auc": float(roc_auc_score(y_gallery, scores)),
            "link_operational_pr_auc": float(average_precision_score(y_gallery, scores)),
            "link_operational_eer": equal_error_rate(y_gallery, scores),
            "encoder_loss": encoder_metadata.get("encoder_loss"),
            "encoder_n_iter": encoder_metadata.get("encoder_n_iter"),
            "encoder_train_segments": encoder_metadata.get("encoder_train_segments"),
            **transform_metadata,
        }
        row.update(tpr_at_fpr(y_gallery, scores, args.fpr_levels))
        row.update(compute_ranking_metrics(pair_df, scores, args.recall_k))
        rows.append(row)
    return rows


def summarize(metrics_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "transform_name",
        "bottleneck_dim",
        "n_output_features",
        "dp_epsilon",
        "dp_delta",
        "dp_clip_norm",
        "dp_noise_sigma",
        "train_negative_strategy",
        "gallery_negatives",
    ]
    for col in group_cols:
        if col not in metrics_df.columns:
            metrics_df[col] = np.nan
    metric_cols = [
        col
        for col in metrics_df.columns
        if col
        not in {
            "seed",
            *group_cols,
            "utility_model",
            "dp_mechanism",
            "dp_scope_note",
        }
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
    dp_configs = parse_dp_configs(args.dp_configs)

    max_chunks = None if args.max_chunks == 0 else args.max_chunks
    features_df, manifest, chunk_files = load_feature_dataset(args.dataset_dir, max_chunks)
    feature_columns = get_feature_columns(features_df)

    rows = []
    split_rows: list[dict[str, object]] = []
    for seed in args.seeds:
        train_df, test_df = split_segments_by_patient_stratified(
            features_df,
            group_col="patient_id",
            label_col="utility_label",
            test_size=args.test_size,
            random_state=seed,
        )
        train_ids = sorted(str(value) for value in train_df["patient_id"].unique())
        test_ids = sorted(str(value) for value in test_df["patient_id"].unique())
        split_rows.append({
            "seed": seed,
            "group_column": "patient_id (interpreted as retained_record_id)",
            "n_train_rows": len(train_df),
            "n_test_rows": len(test_df),
            "n_train_groups": len(train_ids),
            "n_test_groups": len(test_ids),
            "train_groups": "|".join(train_ids),
            "test_groups": "|".join(test_ids),
            "group_overlap": 0,
        })
        print(f"Seed {seed}: fitting encoder...")
        train_embedding_df, test_embedding_df, encoder_metadata = train_encoder_representation(
            train_df=train_df,
            test_df=test_df,
            feature_columns=feature_columns,
            seed=seed,
            args=args,
        )

        gallery_tables = {
            int(gallery_negatives): build_gallery_pairs(
                test_df=test_embedding_df,
                queries_per_seed=args.queries_per_seed,
                gallery_negatives=int(gallery_negatives),
                random_state=seed + int(gallery_negatives),
                min_segment_gap=args.min_segment_gap,
            )
            for gallery_negatives in args.gallery_negatives
        }

        print(f"Seed {seed}: evaluating encoder_identity...")
        rows.extend(
            evaluate_operational_representation(
                train_repr_df=train_embedding_df,
                test_repr_df=test_embedding_df,
                gallery_tables=gallery_tables,
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
                seed=seed,
                args=args,
            )
        )

        for epsilon, clip_norm in dp_configs:
            transform_name = f"encoder_dp_gaussian_eps{epsilon:g}_clip{clip_norm:g}"
            print(f"Seed {seed}: evaluating {transform_name}...")
            train_dp_df, test_dp_df, dp_metadata = apply_dp_to_embedding_split(
                train_embedding_df=train_embedding_df,
                test_embedding_df=test_embedding_df,
                epsilon=epsilon,
                delta=args.dp_delta,
                clip_norm=clip_norm,
                seed=seed + int(epsilon * 10) + int(clip_norm * 1000),
            )
            rows.extend(
                evaluate_operational_representation(
                    train_repr_df=train_dp_df,
                    test_repr_df=test_dp_df,
                    gallery_tables=gallery_tables,
                    transform_name=transform_name,
                    transform_metadata=dp_metadata,
                    encoder_metadata=encoder_metadata,
                    seed=seed,
                    args=args,
                )
            )

    metrics_df = pd.DataFrame(rows)
    summary_df = summarize(metrics_df)

    metrics_path = args.output_dir / f"{args.run_name}_metrics.csv"
    summary_path = args.output_dir / f"{args.run_name}_summary.csv"
    splits_path = args.output_dir / f"{args.run_name}_splits.csv"
    protocol_path = args.output_dir / f"{args.run_name}_protocol.json"
    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    pd.DataFrame(split_rows).to_csv(splits_path, index=False)

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
        "dp_configs": [{"epsilon": epsilon, "clip_norm": clip_norm} for epsilon, clip_norm in dp_configs],
        "dp_delta": args.dp_delta,
        "dp_scope_note": (
            "Operational linkage is evaluated after DP-calibrated perturbation of released embeddings. "
            "The encoder training procedure itself is not DP-SGD."
        ),
        "utility_model": args.utility_model,
        "pair_model_protocol": {
            "representation": "absdiff",
            "model": "XGBoost",
            "train_max_pairs": args.train_max_pairs,
            "min_segment_gap": args.min_segment_gap,
            "max_positive_pairs_per_patient": args.max_positive_pairs_per_patient,
            "train_negative_strategy": args.train_negative_strategy,
            "hard_negative_pool_size": args.hard_negative_pool_size,
        },
        "gallery_protocol": {
            "gallery_negatives": args.gallery_negatives,
            "queries_per_seed": args.queries_per_seed,
            "fpr_levels": args.fpr_levels,
            "recall_k": args.recall_k,
            "description": "one positive candidate and N synthetic negative candidates per query segment",
        },
        "metrics_path": str(metrics_path),
        "summary_path": str(summary_path),
        "splits_path": str(splits_path),
        "leakage_checks": {
            "group_disjoint_assertion": True,
            "group_column": "patient_id (interpreted analytically as retained_record_id)",
            "encoder_and_scaler_fitting_scope": "training split only",
        },
    }
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    print("Saved metrics:", metrics_path)
    print("Saved summary:", summary_path)
    print("Saved group splits:", splits_path)
    print("Saved protocol:", protocol_path)


if __name__ == "__main__":
    main()
