from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

try:
    from config import FINAL_SEGMENT_FEATURES_DIR, OUTPUTS_TABLES_DIR
    from modeling import (
        _build_xgb_model,
        build_pair_table,
        get_feature_columns,
        split_segments_by_patient_stratified,
    )
except ImportError:  # pragma: no cover - package import fallback
    from .config import FINAL_SEGMENT_FEATURES_DIR, OUTPUTS_TABLES_DIR
    from .modeling import (
        _build_xgb_model,
        build_pair_table,
        get_feature_columns,
        split_segments_by_patient_stratified,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate operational-style same-retained-record linkage with synthetic galleries."
    )
    parser.add_argument("--dataset-dir", type=Path, default=FINAL_SEGMENT_FEATURES_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_TABLES_DIR)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--train-max-pairs", type=int, default=2000)
    parser.add_argument("--min-segment-gap", type=int, default=4)
    parser.add_argument("--max-positive-pairs-per-patient", type=int, default=3)
    parser.add_argument("--negative-strategy", choices=["random", "hard"], default="random")
    parser.add_argument("--gallery-negatives", type=int, nargs="+", default=[99, 999])
    parser.add_argument("--queries-per-seed", type=int, default=1000)
    parser.add_argument("--fpr-levels", type=float, nargs="+", default=[0.001, 0.005, 0.01])
    parser.add_argument("--recall-k", type=int, nargs="+", default=[1, 5, 10])
    return parser.parse_args()


def load_feature_dataset(dataset_dir: Path, max_chunks: int | None) -> tuple[pd.DataFrame, dict, list[Path]]:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    chunk_files = [dataset_dir / item["chunk_file"] for item in manifest["chunks"]]
    if max_chunks is not None:
        chunk_files = chunk_files[:max_chunks]
    features_df = pd.concat(
        [pd.read_csv(chunk_file, compression="gzip", low_memory=False) for chunk_file in chunk_files],
        ignore_index=True,
    )
    return features_df, manifest, chunk_files


def get_positive_scores(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        positive_idx = classes.index(1) if 1 in classes else 1
        return model.predict_proba(X)[:, positive_idx]
    return model.decision_function(X)


def build_absdiff_pair_features_fast(
    features_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    feature_matrix = features_df[feature_columns].to_numpy(dtype=float)
    idx_a = pair_df["idx_a"].to_numpy(dtype=int)
    idx_b = pair_df["idx_b"].to_numpy(dtype=int)
    X_pair = np.abs(feature_matrix[idx_a] - feature_matrix[idx_b])
    y_pair = pair_df["pair_label"].to_numpy(dtype=int)
    return X_pair, y_pair


def tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, fpr_levels: list[float]) -> dict[str, float]:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    out = {}
    for level in fpr_levels:
        valid = np.where(fpr <= level)[0]
        out[f"tpr_at_fpr_{level:g}"] = float(np.max(tpr[valid])) if len(valid) else 0.0
    return out


def equal_error_rate(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    crossover_idx = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[crossover_idx] + fnr[crossover_idx]) / 2.0)


def build_gallery_pairs(
    test_df: pd.DataFrame,
    queries_per_seed: int,
    gallery_negatives: int,
    random_state: int,
    min_segment_gap: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    patients = []
    for patient_id, group in test_df.groupby("patient_id"):
        ordered = group.sort_values("segment_ref").reset_index()
        if len(ordered) < 2:
            continue
        possible = []
        records = ordered[["index", "segment_ref"]].to_dict("records")
        for left in records:
            for right in records:
                if int(left["index"]) == int(right["index"]):
                    continue
                if abs(int(left["segment_ref"]) - int(right["segment_ref"])) < min_segment_gap:
                    continue
                possible.append((int(left["index"]), int(right["index"])))
        if possible:
            patients.append((patient_id, possible))

    if not patients:
        raise ValueError("No patients have valid query/positive-gallery segment pairs.")

    patient_ids = [item[0] for item in patients]
    patient_to_indices = {
        patient_id: test_df.index[test_df["patient_id"] == patient_id].to_numpy(dtype=int)
        for patient_id in test_df["patient_id"].unique()
    }
    all_patient_ids = np.asarray(list(patient_to_indices.keys()))

    if queries_per_seed > len(patients):
        selected_positions = rng.choice(len(patients), size=queries_per_seed, replace=True)
    else:
        selected_positions = rng.choice(len(patients), size=queries_per_seed, replace=False)

    pair_rows = []
    query_rows = []
    for query_rank, pos in enumerate(selected_positions):
        patient_id, possible_pairs = patients[int(pos)]
        query_idx, positive_idx = possible_pairs[int(rng.integers(0, len(possible_pairs)))]
        candidate_rows = [{"idx_a": query_idx, "idx_b": positive_idx, "pair_label": 1, "query_id": query_rank}]

        negative_patient_pool = all_patient_ids[all_patient_ids != patient_id]
        replace = gallery_negatives > len(negative_patient_pool)
        negative_patients = rng.choice(negative_patient_pool, size=gallery_negatives, replace=replace)
        for negative_patient in negative_patients:
            negative_indices = patient_to_indices[negative_patient]
            negative_idx = int(rng.choice(negative_indices))
            candidate_rows.append({"idx_a": query_idx, "idx_b": negative_idx, "pair_label": 0, "query_id": query_rank})

        pair_rows.extend(candidate_rows)
        query_rows.append(
            {
                "query_id": query_rank,
                "patient_id": patient_id,
                "query_idx": query_idx,
                "positive_idx": positive_idx,
                "n_candidates": len(candidate_rows),
                "positive_prevalence": 1.0 / len(candidate_rows),
            }
        )

    return pd.DataFrame(pair_rows), pd.DataFrame(query_rows)


def compute_ranking_metrics(pair_df: pd.DataFrame, scores: np.ndarray, recall_k: list[int]) -> dict[str, float]:
    scored = pair_df.copy()
    scored["score"] = scores
    reciprocal_ranks = []
    recall_hits = {k: [] for k in recall_k}

    for _, group in scored.groupby("query_id"):
        ranked = group.sort_values("score", ascending=False).reset_index(drop=True)
        positive_positions = ranked.index[ranked["pair_label"] == 1].to_numpy()
        if len(positive_positions) != 1:
            continue
        rank = int(positive_positions[0]) + 1
        reciprocal_ranks.append(1.0 / rank)
        for k in recall_k:
            recall_hits[k].append(float(rank <= k))

    metrics = {"mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else np.nan}
    for k, hits in recall_hits.items():
        metrics[f"recall_at_{k}"] = float(np.mean(hits)) if hits else np.nan
    return metrics


def evaluate_seed(args, features_df: pd.DataFrame, seed: int) -> list[dict]:
    train_df, test_df = split_segments_by_patient_stratified(
        features_df,
        group_col="patient_id",
        label_col="utility_label",
        test_size=args.test_size,
        random_state=seed,
    )
    feature_columns = get_feature_columns(features_df)

    train_pair_df = build_pair_table(
        train_df,
        max_pairs=args.train_max_pairs,
        random_state=seed,
        min_segment_gap=args.min_segment_gap,
        max_pairs_per_patient=args.max_positive_pairs_per_patient,
        negative_strategy=args.negative_strategy,
    )
    X_train, y_train = build_absdiff_pair_features_fast(train_df, train_pair_df, feature_columns)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    model = _build_xgb_model()
    model.fit(X_train, y_train)

    rows = []
    for gallery_negatives in args.gallery_negatives:
        pair_df, query_df = build_gallery_pairs(
            test_df=test_df,
            queries_per_seed=args.queries_per_seed,
            gallery_negatives=gallery_negatives,
            random_state=seed + gallery_negatives,
            min_segment_gap=args.min_segment_gap,
        )
        X_gallery, y_gallery = build_absdiff_pair_features_fast(test_df, pair_df, feature_columns)
        X_gallery = scaler.transform(X_gallery)
        scores = get_positive_scores(model, X_gallery)

        row = {
            "seed": seed,
            "gallery_negatives": gallery_negatives,
            "queries": len(query_df),
            "candidate_pairs": len(pair_df),
            "positive_prevalence": float(pair_df["pair_label"].mean()),
            "roc_auc": roc_auc_score(y_gallery, scores),
            "pr_auc": average_precision_score(y_gallery, scores),
            "eer": equal_error_rate(y_gallery, scores),
        }
        row.update(tpr_at_fpr(y_gallery, scores, args.fpr_levels))
        row.update(compute_ranking_metrics(pair_df, scores, args.recall_k))
        rows.append(row)

    return rows


def summarize(metrics_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["gallery_negatives"]
    metric_cols = [col for col in metrics_df.columns if col not in {"seed", "gallery_negatives"}]
    summary = metrics_df.groupby(group_cols, as_index=False)[metric_cols].agg(["mean", "std"])
    summary.columns = [
        "_".join([part for part in col if part]) if isinstance(col, tuple) else col for col in summary.columns
    ]
    return summary


def main() -> None:
    args = parse_args()
    started = time.time()
    run_name = args.run_name or f"operational_linkage_{int(started)}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    features_df, manifest, chunk_files = load_feature_dataset(args.dataset_dir, args.max_chunks)

    rows = []
    for seed in args.seeds:
        print(f"Evaluating operational linkage seed {seed}...")
        rows.extend(evaluate_seed(args, features_df, seed))

    metrics_df = pd.DataFrame(rows)
    summary_df = summarize(metrics_df)

    metrics_path = args.output_dir / f"{run_name}_metrics.csv"
    summary_path = args.output_dir / f"{run_name}_summary.csv"
    protocol_path = args.output_dir / f"{run_name}_protocol.json"
    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    protocol = {
        "run_name": run_name,
        "created_at_epoch": started,
        "elapsed_seconds": round(time.time() - started, 2),
        "dataset_dir": str(args.dataset_dir),
        "manifest_total_segments": manifest.get("total_segments"),
        "loaded_chunks": len(chunk_files),
        "max_chunks": args.max_chunks,
        "loaded_segments": len(features_df),
        "seeds": args.seeds,
        "test_size": args.test_size,
        "train_max_pairs": args.train_max_pairs,
        "min_segment_gap": args.min_segment_gap,
        "max_positive_pairs_per_patient": args.max_positive_pairs_per_patient,
        "negative_strategy": args.negative_strategy,
        "gallery_negatives": args.gallery_negatives,
        "queries_per_seed": args.queries_per_seed,
        "fpr_levels": args.fpr_levels,
        "recall_k": args.recall_k,
        "gallery_protocol": "one positive candidate and N synthetic negative candidates per query segment",
        "metrics_path": str(metrics_path),
        "summary_path": str(summary_path),
    }
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    print("Saved metrics:", metrics_path)
    print("Saved summary:", summary_path)
    print("Saved protocol:", protocol_path)


if __name__ == "__main__":
    main()
