from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
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
    from run_operational_linkage_eval import (
        build_absdiff_pair_features_fast,
        build_gallery_pairs,
        compute_ranking_metrics,
        get_positive_scores,
        load_feature_dataset,
        tpr_at_fpr,
    )
    from run_representation_dimension_comparison import build_supervised_encoder_representation
except ImportError:  # pragma: no cover - package import fallback
    from .config import FINAL_SEGMENT_FEATURES_DIR, OUTPUTS_TABLES_DIR
    from .modeling import (
        _build_xgb_model,
        build_pair_table,
        evaluate_utility_model_on_split,
        get_feature_columns,
        split_segments_by_patient_stratified,
    )
    from .run_operational_linkage_eval import (
        build_absdiff_pair_features_fast,
        build_gallery_pairs,
        compute_ranking_metrics,
        get_positive_scores,
        load_feature_dataset,
        tpr_at_fpr,
    )
    from .run_representation_dimension_comparison import build_supervised_encoder_representation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a supervised utility bottleneck encoder and evaluate same-retained-record "
            "linkage with an operational-style synthetic gallery."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, default=FINAL_SEGMENT_FEATURES_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_TABLES_DIR)
    parser.add_argument("--run-name", default="encoder_operational_linkage")
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


def evaluate_seed(args: argparse.Namespace, features_df: pd.DataFrame, seed: int) -> list[dict]:
    train_df, test_df = split_segments_by_patient_stratified(
        features_df,
        group_col="patient_id",
        label_col="utility_label",
        test_size=args.test_size,
        random_state=seed,
    )
    feature_columns = get_feature_columns(features_df)

    train_embedding_df, test_embedding_df, encoder_metadata = build_supervised_encoder_representation(
        train_df=train_df,
        test_df=test_df,
        feature_columns=feature_columns,
        dim=args.bottleneck_dim,
        seed=seed,
        args=args,
    )
    embedding_columns = get_feature_columns(train_embedding_df)

    utility_out = evaluate_utility_model_on_split(
        train_df=train_embedding_df,
        test_df=test_embedding_df,
        model_name=args.utility_model,
        target_col="utility_label",
        feature_columns=embedding_columns,
    )
    utility_eval = utility_out["evaluation"]

    train_pair_df = build_pair_table(
        train_embedding_df,
        max_pairs=args.train_max_pairs,
        random_state=seed,
        min_segment_gap=args.min_segment_gap,
        max_pairs_per_patient=args.max_positive_pairs_per_patient,
        negative_strategy=args.train_negative_strategy,
        hard_negative_pool_size=args.hard_negative_pool_size,
    )
    X_train_pair, y_train_pair = build_absdiff_pair_features_fast(
        train_embedding_df,
        train_pair_df,
        embedding_columns,
    )

    pair_scaler = StandardScaler()
    X_train_pair = pair_scaler.fit_transform(X_train_pair)
    link_model = _build_xgb_model()
    link_model.fit(X_train_pair, y_train_pair)

    rows = []
    direct = encoder_metadata.get("encoder_direct", {})
    for gallery_negatives in args.gallery_negatives:
        pair_df, query_df = build_gallery_pairs(
            test_df=test_embedding_df,
            queries_per_seed=args.queries_per_seed,
            gallery_negatives=gallery_negatives,
            random_state=seed + gallery_negatives,
            min_segment_gap=args.min_segment_gap,
        )
        X_gallery, y_gallery = build_absdiff_pair_features_fast(test_embedding_df, pair_df, embedding_columns)
        X_gallery = pair_scaler.transform(X_gallery)
        scores = get_positive_scores(link_model, X_gallery)

        row = {
            "seed": seed,
            "bottleneck_dim": int(args.bottleneck_dim),
            "n_embedding_features": len(embedding_columns),
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
            "link_operational_roc_auc": float(roc_auc_score(y_gallery, scores)),
            "link_operational_pr_auc": float(average_precision_score(y_gallery, scores)),
            "encoder_loss": encoder_metadata.get("encoder_loss"),
            "encoder_n_iter": encoder_metadata.get("encoder_n_iter"),
            "encoder_train_segments": encoder_metadata.get("encoder_train_segments"),
        }
        row.update(tpr_at_fpr(y_gallery, scores, args.fpr_levels))
        row.update(compute_ranking_metrics(pair_df, scores, args.recall_k))
        rows.append(row)

    return rows


def summarize(metrics_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["bottleneck_dim", "train_negative_strategy", "gallery_negatives"]
    metric_cols = [col for col in metrics_df.columns if col not in {"seed", *group_cols, "utility_model"}]
    summary = metrics_df.groupby(group_cols, as_index=False)[metric_cols].agg(["mean", "std"])
    summary.columns = [
        "_".join([part for part in col if part]) if isinstance(col, tuple) else col for col in summary.columns
    ]
    return summary


def main() -> None:
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    max_chunks = None if args.max_chunks == 0 else args.max_chunks
    features_df, manifest, chunk_files = load_feature_dataset(args.dataset_dir, max_chunks)

    rows = []
    for seed in args.seeds:
        print(f"Evaluating encoder operational linkage seed {seed}...")
        rows.extend(evaluate_seed(args, features_df, seed))

    metrics_df = pd.DataFrame(rows)
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
        "encoder_batch_size": args.encoder_batch_size,
        "encoder_learning_rate_init": args.encoder_learning_rate_init,
        "encoder_train_max_segments": args.encoder_train_max_segments,
        "utility_model": args.utility_model,
        "train_max_pairs": args.train_max_pairs,
        "min_segment_gap": args.min_segment_gap,
        "max_positive_pairs_per_patient": args.max_positive_pairs_per_patient,
        "train_negative_strategy": args.train_negative_strategy,
        "hard_negative_pool_size": args.hard_negative_pool_size,
        "gallery_negatives": args.gallery_negatives,
        "queries_per_seed": args.queries_per_seed,
        "fpr_levels": args.fpr_levels,
        "recall_k": args.recall_k,
        "gallery_protocol": "one positive candidate and N synthetic negative candidates per query segment",
        "encoder_protocol": (
            "supervised utility MLP fitted on train split only; final hidden layer used as bottleneck embedding"
        ),
        "pair_model_protocol": "XGBoost on absolute differences between embedding vectors",
        "metrics_path": str(metrics_path),
        "summary_path": str(summary_path),
    }
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    print("Saved metrics:", metrics_path)
    print("Saved summary:", summary_path)
    print("Saved protocol:", protocol_path)


if __name__ == "__main__":
    main()
