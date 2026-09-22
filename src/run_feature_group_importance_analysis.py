from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

try:
    from config import FINAL_SEGMENT_FEATURES_DIR, OUTPUTS_TABLES_DIR
    from modeling import (
        _extract_model_based_importance,
        fit_linkability_model,
        fit_utility_model,
        get_feature_columns,
    )
except ImportError:  # pragma: no cover - package import fallback
    from .config import FINAL_SEGMENT_FEATURES_DIR, OUTPUTS_TABLES_DIR
    from .modeling import (
        _extract_model_based_importance,
        fit_linkability_model,
        fit_utility_model,
        get_feature_columns,
    )


STAT_SUFFIXES = [
    "zero_crossing_rate",
    "abs_mean",
    "n_peaks",
    "amplitude",
    "skewness",
    "kurtosis",
    "energy",
    "area",
    "mean",
    "std",
    "min",
    "max",
    "rms",
]

STAT_GROUPS = {
    "mean": "central_tendency",
    "std": "variability",
    "min": "amplitude_magnitude",
    "max": "amplitude_magnitude",
    "amplitude": "amplitude_magnitude",
    "energy": "amplitude_magnitude",
    "area": "amplitude_magnitude",
    "abs_mean": "amplitude_magnitude",
    "rms": "amplitude_magnitude",
    "zero_crossing_rate": "temporal_variability",
    "n_peaks": "temporal_variability",
    "skewness": "shape_distribution",
    "kurtosis": "shape_distribution",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare utility and linkability feature importances by interpretable feature groups."
    )
    parser.add_argument("--dataset-dir", type=Path, default=FINAL_SEGMENT_FEATURES_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_TABLES_DIR)
    parser.add_argument("--run-name", default="feature_group_importance")
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=20,
        help="Number of final dataset chunks to load. Use 0 to load all chunks.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--utility-model", default="LogisticRegression")
    parser.add_argument("--linkability-model", default="XGBoost")
    parser.add_argument("--link-max-pairs", type=int, default=2000)
    parser.add_argument("--link-min-segment-gap", type=int, default=4)
    parser.add_argument("--link-max-positive-pairs-per-patient", type=int, default=3)
    parser.add_argument("--negative-strategy", choices=["random", "hard"], default="random")
    parser.add_argument("--hard-negative-pool-size", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=30)
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


def strip_pair_prefix(feature_name: str) -> str:
    for prefix in ("absdiff__", "mean__"):
        if feature_name.startswith(prefix):
            return feature_name[len(prefix) :]
    return feature_name


def split_stat_suffix(name: str) -> tuple[str, str]:
    for suffix in STAT_SUFFIXES:
        token = f"_{suffix}"
        if name.endswith(token):
            return name[: -len(token)], suffix
    return name, "unknown"


def parse_feature_name(feature_name: str) -> dict[str, str]:
    base_feature = strip_pair_prefix(feature_name)

    if base_feature.startswith("lead_"):
        remainder = base_feature[len("lead_") :]
        lead_name, statistic = split_stat_suffix(remainder)
        return {
            "feature": base_feature,
            "feature_scope": "lead",
            "lead_or_global": lead_name,
            "global_aggregation": "",
            "feature_stat": statistic,
            "stat_group": STAT_GROUPS.get(statistic, "other"),
        }

    if base_feature.startswith("global_"):
        remainder = base_feature[len("global_") :]
        aggregation, statistic = split_stat_suffix(remainder)
        return {
            "feature": base_feature,
            "feature_scope": "global",
            "lead_or_global": "global",
            "global_aggregation": aggregation,
            "feature_stat": statistic,
            "stat_group": STAT_GROUPS.get(statistic, "other"),
        }

    return {
        "feature": base_feature,
        "feature_scope": "other",
        "lead_or_global": "other",
        "global_aggregation": "",
        "feature_stat": "unknown",
        "stat_group": "other",
    }


def normalize_importance(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    total = out["importance"].sum()
    out["importance_share"] = out["importance"] / total if total > 0 else 0.0
    return out


def utility_importance(features_df: pd.DataFrame, model_name: str, test_size: float, random_state: int) -> pd.DataFrame:
    fitted = fit_utility_model(
        features_df=features_df,
        model_name=model_name,
        test_size=test_size,
        random_state=random_state,
        stratified=True,
    )
    importance_df = _extract_model_based_importance(fitted["model"], fitted["feature_columns"])
    importance_df = normalize_importance(importance_df)
    importance_df.insert(0, "task", "utility")
    importance_df.insert(1, "model", model_name)
    return importance_df


def linkability_importance(
    features_df: pd.DataFrame,
    model_name: str,
    test_size: float,
    random_state: int,
    max_pairs: int,
    min_segment_gap: int,
    max_positive_pairs_per_patient: int,
    negative_strategy: str,
    hard_negative_pool_size: int,
) -> pd.DataFrame:
    fitted = fit_linkability_model(
        features_df=features_df,
        model_name=model_name,
        test_size=test_size,
        random_state=random_state,
        max_pairs=max_pairs,
        representation="absdiff",
        min_segment_gap=min_segment_gap,
        max_positive_pairs_per_patient=max_positive_pairs_per_patient,
        negative_strategy=negative_strategy,
        hard_negative_pool_size=hard_negative_pool_size,
    )
    importance_df = _extract_model_based_importance(fitted["model"], fitted["pair_feature_names"])
    importance_df["feature"] = importance_df["feature"].map(strip_pair_prefix)
    importance_df = importance_df.groupby("feature", as_index=False)["importance"].sum()
    importance_df = normalize_importance(importance_df)
    importance_df.insert(0, "task", "linkability")
    importance_df.insert(1, "model", model_name)
    return importance_df


def add_feature_metadata(importance_df: pd.DataFrame) -> pd.DataFrame:
    metadata_df = pd.DataFrame([parse_feature_name(feature) for feature in importance_df["feature"]])
    return pd.concat([importance_df.reset_index(drop=True), metadata_df.drop(columns=["feature"])], axis=1)


def build_group_summary(importance_df: pd.DataFrame) -> pd.DataFrame:
    group_frames = []
    for group_type, column in [
        ("scope", "feature_scope"),
        ("lead_or_global", "lead_or_global"),
        ("statistic", "feature_stat"),
        ("stat_group", "stat_group"),
        ("global_aggregation", "global_aggregation"),
    ]:
        grouped = (
            importance_df.groupby(["task", "model", column], as_index=False)
            .agg(
                importance_share=("importance_share", "sum"),
                raw_importance=("importance", "sum"),
                n_features=("feature", "nunique"),
            )
            .rename(columns={column: "group_value"})
        )
        grouped.insert(0, "group_type", group_type)
        group_frames.append(grouped)
    return pd.concat(group_frames, ignore_index=True)


def build_feature_overlap(importance_df: pd.DataFrame) -> pd.DataFrame:
    metadata_cols = ["feature", "feature_scope", "lead_or_global", "global_aggregation", "feature_stat", "stat_group"]
    metadata = importance_df[metadata_cols].drop_duplicates("feature")
    pivot = (
        importance_df.pivot_table(index="feature", columns="task", values="importance_share", aggfunc="sum")
        .fillna(0.0)
        .reset_index()
    )
    if "utility" not in pivot.columns:
        pivot["utility"] = 0.0
    if "linkability" not in pivot.columns:
        pivot["linkability"] = 0.0
    pivot = pivot.merge(metadata, on="feature", how="left")
    pivot["shared_min_importance"] = pivot[["utility", "linkability"]].min(axis=1)
    pivot["linkability_minus_utility"] = pivot["linkability"] - pivot["utility"]
    pivot["utility_minus_linkability"] = pivot["utility"] - pivot["linkability"]
    pivot["linkability_to_utility_ratio"] = pivot["linkability"] / pivot["utility"].replace(0, pd.NA)
    return pivot.sort_values("shared_min_importance", ascending=False)


def build_candidate_table(overlap_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    frames = []
    selections = [
        ("shared_high", "shared_min_importance", False),
        ("linkability_dominant", "linkability_minus_utility", False),
        ("utility_dominant", "utility_minus_linkability", False),
    ]
    for candidate_type, sort_col, ascending in selections:
        frame = overlap_df.sort_values(sort_col, ascending=ascending).head(top_n).copy()
        frame.insert(0, "candidate_type", candidate_type)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    max_chunks = None if args.max_chunks == 0 else args.max_chunks
    features_df, manifest, chunk_files = load_feature_dataset(args.dataset_dir, max_chunks=max_chunks)
    feature_columns = get_feature_columns(features_df)

    utility_df = utility_importance(
        features_df=features_df,
        model_name=args.utility_model,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    linkability_df = linkability_importance(
        features_df=features_df,
        model_name=args.linkability_model,
        test_size=args.test_size,
        random_state=args.random_state,
        max_pairs=args.link_max_pairs,
        min_segment_gap=args.link_min_segment_gap,
        max_positive_pairs_per_patient=args.link_max_positive_pairs_per_patient,
        negative_strategy=args.negative_strategy,
        hard_negative_pool_size=args.hard_negative_pool_size,
    )

    importance_df = add_feature_metadata(pd.concat([utility_df, linkability_df], ignore_index=True))
    group_summary_df = build_group_summary(importance_df)
    overlap_df = build_feature_overlap(importance_df)
    candidates_df = build_candidate_table(overlap_df, top_n=args.top_n)

    importance_path = args.output_dir / f"{args.run_name}_feature_importance_long.csv"
    group_path = args.output_dir / f"{args.run_name}_group_summary.csv"
    overlap_path = args.output_dir / f"{args.run_name}_feature_overlap.csv"
    candidates_path = args.output_dir / f"{args.run_name}_customization_candidates.csv"
    protocol_path = args.output_dir / f"{args.run_name}_protocol.json"

    importance_df.to_csv(importance_path, index=False)
    group_summary_df.to_csv(group_path, index=False)
    overlap_df.to_csv(overlap_path, index=False)
    candidates_df.to_csv(candidates_path, index=False)

    protocol = {
        "run_name": args.run_name,
        "created_at_epoch": started,
        "elapsed_seconds": round(time.time() - started, 2),
        "dataset_dir": str(args.dataset_dir),
        "manifest_total_segments": manifest.get("total_segments"),
        "loaded_chunks": len(chunk_files),
        "max_chunks": max_chunks,
        "loaded_segments": len(features_df),
        "loaded_identifiers": int(features_df["patient_id"].nunique()) if "patient_id" in features_df else None,
        "n_features": len(feature_columns),
        "test_size": args.test_size,
        "random_state": args.random_state,
        "utility_model": args.utility_model,
        "linkability_model": args.linkability_model,
        "linkability_protocol": {
            "representation": "absdiff",
            "max_pairs": args.link_max_pairs,
            "min_segment_gap": args.link_min_segment_gap,
            "max_positive_pairs_per_patient": args.link_max_positive_pairs_per_patient,
            "negative_strategy": args.negative_strategy,
            "hard_negative_pool_size": args.hard_negative_pool_size,
        },
        "importance_type": "model-based importance; absolute coefficients for LogisticRegression and feature_importances_ for XGBoost",
        "outputs": {
            "feature_importance_long": str(importance_path),
            "group_summary": str(group_path),
            "feature_overlap": str(overlap_path),
            "customization_candidates": str(candidates_path),
        },
    }
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    print("Saved feature importances:", importance_path)
    print("Saved group summary:", group_path)
    print("Saved feature overlap:", overlap_path)
    print("Saved customization candidates:", candidates_path)
    print("Saved protocol:", protocol_path)


if __name__ == "__main__":
    main()
