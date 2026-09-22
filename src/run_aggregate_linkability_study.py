from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from config import FINAL_SEGMENT_FEATURES_DIR, OUTPUTS_TABLES_DIR
    from modeling import (
        evaluate_utility_model_on_split,
        get_feature_columns,
        run_linkability_baselines_on_split,
        split_segments_by_patient_stratified,
    )
except ImportError:  # pragma: no cover - package import fallback
    from .config import FINAL_SEGMENT_FEATURES_DIR, OUTPUTS_TABLES_DIR
    from .modeling import (
        evaluate_utility_model_on_split,
        get_feature_columns,
        run_linkability_baselines_on_split,
        split_segments_by_patient_stratified,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate utility and linkability after aggregating multiple ECG segments into record-level views."
    )
    parser.add_argument("--dataset-dir", type=Path, default=FINAL_SEGMENT_FEATURES_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_TABLES_DIR)
    parser.add_argument("--run-name", default="aggregate_linkability")
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=20,
        help="Number of final dataset chunks to load. Use 0 to load all chunks.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--protocols",
        nargs="+",
        default=["odd_even", "early_late_gap", "random_halves"],
        choices=["odd_even", "early_late_gap", "random_halves"],
    )
    parser.add_argument(
        "--stats",
        nargs="+",
        default=["mean", "mean_std"],
        choices=["mean", "mean_std"],
        help="Aggregation feature representation.",
    )
    parser.add_argument("--utility-model", default="LogisticRegression")
    parser.add_argument(
        "--skip-utility",
        action="store_true",
        help="Skip aggregate utility evaluation and only run aggregate linkability.",
    )
    parser.add_argument("--link-max-pairs", type=int, default=2000)
    parser.add_argument("--negative-strategy", choices=["random", "hard"], default="random")
    parser.add_argument("--hard-negative-pool-size", type=int, default=5)
    parser.add_argument("--include-distance-baselines", action="store_true")
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


def stable_patient_seed(patient_id: str, random_state: int) -> int:
    patient_hash = zlib.crc32(str(patient_id).encode("utf-8"))
    return int((patient_hash + random_state) % (2**32 - 1))


def split_segment_refs(segment_refs: list[int], protocol: str, patient_id: str, random_state: int) -> dict[str, list[int]]:
    refs = sorted(int(ref) for ref in segment_refs)

    if protocol == "odd_even":
        return {
            "even": [ref for ref in refs if ref % 2 == 0],
            "odd": [ref for ref in refs if ref % 2 == 1],
        }

    if protocol == "early_late_gap":
        return {
            "early": [ref for ref in refs if ref <= 3],
            "late": [ref for ref in refs if ref >= 5],
        }

    if protocol == "random_halves":
        rng = np.random.default_rng(stable_patient_seed(patient_id, random_state))
        shuffled = refs.copy()
        rng.shuffle(shuffled)
        split_at = len(shuffled) // 2
        return {
            "random_a": sorted(shuffled[:split_at]),
            "random_b": sorted(shuffled[split_at:]),
        }

    raise ValueError(f"Unsupported aggregate protocol: {protocol}")


def aggregate_feature_block(block: pd.DataFrame, feature_columns: list[str], stats: str) -> dict[str, float]:
    values = block[feature_columns].apply(pd.to_numeric, errors="coerce")
    row: dict[str, float] = {}

    means = values.mean(axis=0)
    if stats == "mean":
        row.update({f"agg_mean__{feature}": float(value) for feature, value in means.items()})
        return row

    if stats == "mean_std":
        stds = values.std(axis=0, ddof=0).fillna(0.0)
        row.update({f"agg_mean__{feature}": float(value) for feature, value in means.items()})
        row.update({f"agg_std__{feature}": float(value) for feature, value in stds.items()})
        return row

    raise ValueError("stats must be 'mean' or 'mean_std'.")


def build_aggregate_views(
    segment_df: pd.DataFrame,
    feature_columns: list[str],
    protocol: str,
    stats: str,
    random_state: int,
) -> pd.DataFrame:
    if protocol in {"odd_even", "early_late_gap"}:
        return build_aggregate_views_vectorized(
            segment_df=segment_df,
            feature_columns=feature_columns,
            protocol=protocol,
            stats=stats,
        )

    rows = []

    for patient_id, group in segment_df.groupby("patient_id", sort=False):
        group = group.sort_values("segment_ref")
        split_map = split_segment_refs(
            group["segment_ref"].tolist(),
            protocol=protocol,
            patient_id=str(patient_id),
            random_state=random_state,
        )

        for view_index, (view_name, refs) in enumerate(split_map.items()):
            if not refs:
                continue
            block = group[group["segment_ref"].isin(refs)]
            if block.empty:
                continue

            label_values = block["label"].dropna().astype(str)
            row = {
                "patient_id": patient_id,
                "segment_id": f"{patient_id}_{protocol}_{view_name}",
                "label": label_values.iloc[0] if len(label_values) else None,
                "utility_label": block["utility_label"].iloc[0],
                "segment_ref": view_index,
                "start_sample": int(block["start_sample"].min()) if "start_sample" in block else view_index,
                "end_sample": int(block["end_sample"].max()) if "end_sample" in block else view_index,
            }
            row.update(aggregate_feature_block(block, feature_columns=feature_columns, stats=stats))
            rows.append(row)

    return pd.DataFrame(rows)


def build_aggregate_views_vectorized(
    segment_df: pd.DataFrame,
    feature_columns: list[str],
    protocol: str,
    stats: str,
) -> pd.DataFrame:
    frame = segment_df.copy()

    if protocol == "odd_even":
        frame["aggregate_view"] = np.where(frame["segment_ref"].astype(int) % 2 == 0, "even", "odd")
        view_index = {"even": 0, "odd": 1}
    elif protocol == "early_late_gap":
        refs = frame["segment_ref"].astype(int)
        frame = frame[(refs <= 3) | (refs >= 5)].copy()
        frame["aggregate_view"] = np.where(frame["segment_ref"].astype(int) <= 3, "early", "late")
        view_index = {"early": 0, "late": 1}
    else:
        raise ValueError(f"Unsupported vectorized aggregate protocol: {protocol}")

    group_cols = ["patient_id", "aggregate_view"]
    metadata = (
        frame.groupby(group_cols, as_index=False)
        .agg(
            label=("label", "first"),
            utility_label=("utility_label", "first"),
            start_sample=("start_sample", "min"),
            end_sample=("end_sample", "max"),
        )
    )
    metadata["segment_ref"] = metadata["aggregate_view"].map(view_index).astype(int)
    metadata["segment_id"] = (
        metadata["patient_id"].astype(str) + f"_{protocol}_" + metadata["aggregate_view"].astype(str)
    )

    if stats == "mean":
        feature_part = frame.groupby(group_cols)[feature_columns].mean().reset_index()
        feature_part = feature_part.rename(columns={feature: f"agg_mean__{feature}" for feature in feature_columns})
    elif stats == "mean_std":
        mean_part = frame.groupby(group_cols)[feature_columns].mean()
        std_part = frame.groupby(group_cols)[feature_columns].std(ddof=0).fillna(0.0)
        mean_part.columns = [f"agg_mean__{feature}" for feature in feature_columns]
        std_part.columns = [f"agg_std__{feature}" for feature in feature_columns]
        feature_part = pd.concat([mean_part, std_part], axis=1).reset_index()
    else:
        raise ValueError("stats must be 'mean' or 'mean_std'.")

    out = metadata.merge(feature_part, on=group_cols, how="inner")
    out = out.sort_values(["patient_id", "segment_ref"]).reset_index(drop=True)
    metadata_cols = ["patient_id", "segment_id", "label", "utility_label", "segment_ref", "start_sample", "end_sample"]
    feature_cols = [col for col in out.columns if col.startswith("agg_")]
    return out[metadata_cols + feature_cols]


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
    summary = metrics_df.groupby(
        ["aggregate_protocol", "aggregate_stats", "linkability_model"],
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
    aggregate_shape_rows = []

    for seed in args.seeds:
        train_segments_df, test_segments_df = split_segments_by_patient_stratified(
            features_df,
            group_col="patient_id",
            label_col="utility_label",
            test_size=args.test_size,
            random_state=seed,
        )

        for protocol in args.protocols:
            for stats in args.stats:
                print(f"Seed {seed}: aggregating protocol={protocol}, stats={stats}...")
                train_views_df = build_aggregate_views(
                    train_segments_df,
                    feature_columns=feature_columns,
                    protocol=protocol,
                    stats=stats,
                    random_state=seed,
                )
                test_views_df = build_aggregate_views(
                    test_segments_df,
                    feature_columns=feature_columns,
                    protocol=protocol,
                    stats=stats,
                    random_state=seed,
                )

                aggregate_shape_rows.append(
                    {
                        "seed": seed,
                        "aggregate_protocol": protocol,
                        "aggregate_stats": stats,
                        "train_rows": len(train_views_df),
                        "test_rows": len(test_views_df),
                        "train_patients": train_views_df["patient_id"].nunique(),
                        "test_patients": test_views_df["patient_id"].nunique(),
                        "n_features": len(get_feature_columns(train_views_df)),
                    }
                )

                if args.skip_utility:
                    utility_eval = {
                        "model": "not_evaluated",
                        "f1_score": np.nan,
                        "balanced_accuracy": np.nan,
                        "roc_auc": np.nan,
                        "pr_auc": np.nan,
                    }
                else:
                    utility_out = evaluate_utility_model_on_split(
                        train_df=train_views_df,
                        test_df=test_views_df,
                        model_name=args.utility_model,
                        target_col="utility_label",
                    )
                    utility_eval = utility_out["evaluation"]

                linkability_out = run_linkability_baselines_on_split(
                    train_df=train_views_df,
                    test_df=test_views_df,
                    max_pairs=args.link_max_pairs,
                    representation="absdiff",
                    random_state=seed,
                    min_segment_gap=1,
                    max_positive_pairs_per_patient=1,
                    negative_strategy=args.negative_strategy,
                    hard_negative_pool_size=args.hard_negative_pool_size,
                    include_distance_baselines=args.include_distance_baselines,
                )

                for _, link_row in linkability_out["summary_df"].iterrows():
                    metrics_rows.append(
                        {
                            "run_name": args.run_name,
                            "seed": seed,
                            "aggregate_protocol": protocol,
                            "aggregate_stats": stats,
                            "train_views": len(train_views_df),
                            "test_views": len(test_views_df),
                            "train_patients": train_views_df["patient_id"].nunique(),
                            "test_patients": test_views_df["patient_id"].nunique(),
                            "n_aggregate_features": len(get_feature_columns(train_views_df)),
                            "utility_model": utility_eval["model"],
                            "utility_f1": utility_eval["f1_score"],
                            "utility_balanced_accuracy": utility_eval["balanced_accuracy"],
                            "utility_roc_auc": utility_eval["roc_auc"],
                            "utility_pr_auc": utility_eval["pr_auc"],
                            "linkability_model": link_row["model"],
                            "linkability_f1": link_row["f1_score"],
                            "linkability_balanced_accuracy": link_row["balanced_accuracy"],
                            "linkability_roc_auc": link_row["roc_auc"],
                            "linkability_pr_auc": link_row["pr_auc"],
                            "link_train_pairs": len(linkability_out["train_pair_df"]),
                            "link_test_pairs": len(linkability_out["test_pair_df"]),
                            "link_train_positive_prevalence": float(
                                linkability_out["train_pair_df"]["pair_label"].mean()
                            ),
                            "link_test_positive_prevalence": float(
                                linkability_out["test_pair_df"]["pair_label"].mean()
                            ),
                        }
                    )

    metrics_df = pd.DataFrame(metrics_rows)
    summary_df = summarize_metrics(metrics_df)
    shape_df = pd.DataFrame(aggregate_shape_rows)

    metrics_path = args.output_dir / f"{args.run_name}_metrics.csv"
    summary_path = args.output_dir / f"{args.run_name}_summary.csv"
    shape_path = args.output_dir / f"{args.run_name}_aggregate_shapes.csv"
    protocol_path = args.output_dir / f"{args.run_name}_protocol.json"

    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    shape_df.to_csv(shape_path, index=False)

    protocol_doc = {
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
        "protocols": args.protocols,
        "stats": args.stats,
        "utility_protocol": {
            "split": "patient-level stratified split before aggregation",
            "target": "utility_label",
            "model": args.utility_model,
            "unit": "aggregate view built from multiple ECG segments",
            "evaluated": not args.skip_utility,
        },
        "linkability_protocol": {
            "unit": "pair of aggregate views",
            "positive_pair": "two disjoint aggregate views from the same retained identifier/record",
            "negative_pair": "aggregate views from different retained identifiers/records",
            "representation": "absolute feature difference between aggregate views",
            "max_pairs": args.link_max_pairs,
            "max_positive_pairs_per_patient": 1,
            "negative_strategy": args.negative_strategy,
            "hard_negative_pool_size": args.hard_negative_pool_size,
            "pair_prevalence": "balanced by construction",
        },
        "aggregation_protocol_details": {
            "odd_even": "view A uses even segment_ref values; view B uses odd segment_ref values",
            "early_late_gap": "view A uses segment_ref <= 3; view B uses segment_ref >= 5; middle segment_ref 4 is excluded",
            "random_halves": "segment_ref values are split into two deterministic random halves per identifier and seed",
        },
        "outputs": {
            "metrics": str(metrics_path),
            "summary": str(summary_path),
            "aggregate_shapes": str(shape_path),
        },
    }
    protocol_path.write_text(json.dumps(protocol_doc, indent=2), encoding="utf-8")

    print("Saved metrics:", metrics_path)
    print("Saved summary:", summary_path)
    print("Saved aggregate shapes:", shape_path)
    print("Saved protocol:", protocol_path)


if __name__ == "__main__":
    main()
