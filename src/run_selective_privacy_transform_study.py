from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import StandardScaler

try:
    from config import FINAL_SEGMENT_FEATURES_DIR, OUTPUTS_TABLES_DIR
    from modeling import (
        METADATA_COLUMNS,
        evaluate_utility_model_on_split,
        get_feature_columns,
        run_linkability_baselines_on_split,
        split_segments_by_patient_stratified,
    )
except ImportError:  # pragma: no cover - package import fallback
    from .config import FINAL_SEGMENT_FEATURES_DIR, OUTPUTS_TABLES_DIR
    from .modeling import (
        METADATA_COLUMNS,
        evaluate_utility_model_on_split,
        get_feature_columns,
        run_linkability_baselines_on_split,
        split_segments_by_patient_stratified,
    )


@dataclass
class TierStats:
    features: list[str]
    medians: pd.Series
    stds: pd.Series
    lower_bounds: pd.Series | None = None
    upper_bounds: pd.Series | None = None


@dataclass
class SelectiveTransform:
    name: str
    method: str
    feature_columns: list[str]
    output_feature_names: list[str]
    linkability_features: list[str]
    shared_features: list[str]
    preserved_utility_features: list[str]
    linkability_noise_fraction: float = 0.0
    shared_noise_fraction: float = 0.0
    winsor_lower_quantile: float | None = None
    winsor_upper_quantile: float | None = None
    random_state: int = 42
    linkability_tier_stats: TierStats | None = None
    shared_tier_stats: TierStats | None = None
    pca_features: list[str] | None = None
    pca_imputer: SimpleImputer | None = None
    pca_scaler: StandardScaler | None = None
    pca: PCA | None = None
    protected_rest_features: list[str] | None = None
    rest_imputer: SimpleImputer | None = None
    rest_scaler: StandardScaler | None = None
    rest_pca: PCA | None = None
    rest_projector: GaussianRandomProjection | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate selective privacy transformations guided by utility/linkability feature importance."
    )
    parser.add_argument("--dataset-dir", type=Path, default=FINAL_SEGMENT_FEATURES_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_TABLES_DIR)
    parser.add_argument(
        "--candidates-path",
        type=Path,
        default=OUTPUTS_TABLES_DIR / "feature_group_importance_full_customization_candidates.csv",
    )
    parser.add_argument("--run-name", default="selective_privacy_transform_full")
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help="Number of final dataset chunks to load. Use 0 to load all chunks.",
    )
    parser.add_argument("--top-per-type", type=int, default=30)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--utility-model", default="LogisticRegression")
    parser.add_argument("--link-max-pairs", type=int, default=2000)
    parser.add_argument("--link-min-segment-gap", type=int, default=4)
    parser.add_argument("--link-max-positive-pairs-per-patient", type=int, default=3)
    parser.add_argument("--negative-strategy", choices=["random", "hard"], default="random")
    parser.add_argument("--hard-negative-pool-size", type=int, default=5)
    parser.add_argument("--pca-components", type=int, nargs="+", default=[10, 15])
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Optional subset of transform names to run, e.g. identity link_pca_10 link_pca_15.",
    )
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


def unique_existing_features(frame: pd.DataFrame, feature_columns: list[str], top_n: int) -> list[str]:
    available = set(feature_columns)
    out: list[str] = []
    for feature in frame["feature"].dropna().astype(str):
        if feature in available and feature not in out:
            out.append(feature)
        if len(out) >= top_n:
            break
    return out


def build_feature_policy(
    candidates_df: pd.DataFrame,
    feature_columns: list[str],
    top_per_type: int,
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    raw_policy = {}
    policy_rows = []
    for candidate_type in ["linkability_dominant", "shared_high", "utility_dominant"]:
        frame = candidates_df[candidates_df["candidate_type"] == candidate_type]
        features = unique_existing_features(frame, feature_columns, top_per_type)
        raw_policy[candidate_type] = features
        policy_rows.append(frame[frame["feature"].isin(features)].copy())

    utility_preserve = set(raw_policy["utility_dominant"])
    linkability_transform = [f for f in raw_policy["linkability_dominant"] if f not in utility_preserve]
    linkability_set = set(linkability_transform)
    shared_light_transform = [
        f for f in raw_policy["shared_high"] if f not in utility_preserve and f not in linkability_set
    ]

    policy = {
        "linkability_dominant": linkability_transform,
        "shared_high": shared_light_transform,
        "utility_dominant": raw_policy["utility_dominant"],
    }

    policy_df = pd.concat(policy_rows, ignore_index=True)
    policy_df["selected_for_transformation"] = policy_df.apply(
        lambda row: (
            row["feature"] in policy["linkability_dominant"]
            or row["feature"] in policy["shared_high"]
        ),
        axis=1,
    )
    policy_df["preserved_as_utility_dominant"] = policy_df["feature"].isin(policy["utility_dominant"])
    return policy, policy_df


def get_selective_transform_suite(seed: int, pca_components: list[int]) -> list[dict]:
    suite = [
        {
            "name": "identity",
            "method": "identity",
            "params": {},
        },
        {
            "name": "link_noise_005",
            "method": "selective_noise",
            "params": {"linkability_noise_fraction": 0.05},
        },
        {
            "name": "link_noise_010",
            "method": "selective_noise",
            "params": {"linkability_noise_fraction": 0.10},
        },
        {
            "name": "link_noise_020",
            "method": "selective_noise",
            "params": {"linkability_noise_fraction": 0.20},
        },
        {
            "name": "tiered_noise_l010_s003",
            "method": "tiered_noise",
            "params": {"linkability_noise_fraction": 0.10, "shared_noise_fraction": 0.03},
        },
        {
            "name": "tiered_noise_l020_s005",
            "method": "tiered_noise",
            "params": {"linkability_noise_fraction": 0.20, "shared_noise_fraction": 0.05},
        },
        {
            "name": "link_winsor05_noise010",
            "method": "selective_winsor_noise",
            "params": {
                "linkability_noise_fraction": 0.10,
                "winsor_lower_quantile": 0.05,
                "winsor_upper_quantile": 0.95,
            },
        },
        {
            "name": "tiered_winsor05_noise_l010_s003",
            "method": "tiered_winsor_noise",
            "params": {
                "linkability_noise_fraction": 0.10,
                "shared_noise_fraction": 0.03,
                "winsor_lower_quantile": 0.05,
                "winsor_upper_quantile": 0.95,
            },
        },
    ]

    for n_components in pca_components:
        suite.append(
            {
                "name": f"link_pca_{n_components}",
                "method": "selective_pca",
                "params": {"pca_components": int(n_components), "random_state": seed},
            }
        )
        suite.append(
            {
                "name": f"link_pca_{n_components}_shared_noise003",
                "method": "selective_pca_tiered",
                "params": {
                    "pca_components": int(n_components),
                    "shared_noise_fraction": 0.03,
                    "random_state": seed,
                },
            }
        )
        suite.append(
            {
                "name": f"preserve_utility_pca_rest_{n_components}",
                "method": "preserve_utility_pca_rest",
                "params": {"n_components": int(n_components), "random_state": seed},
            }
        )
        suite.append(
            {
                "name": f"preserve_utility_rp_rest_{n_components}",
                "method": "preserve_utility_rp_rest",
                "params": {"n_components": int(n_components), "random_state": seed},
            }
        )
    return suite


def _fit_tier_stats(
    train_df: pd.DataFrame,
    features: list[str],
    lower_quantile: float | None = None,
    upper_quantile: float | None = None,
) -> TierStats | None:
    if not features:
        return None

    matrix = train_df[features].apply(pd.to_numeric, errors="coerce")
    medians = matrix.median(axis=0)
    stds = matrix.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
    lower_bounds = None
    upper_bounds = None
    if lower_quantile is not None and upper_quantile is not None:
        lower_bounds = matrix.quantile(lower_quantile, axis=0)
        upper_bounds = matrix.quantile(upper_quantile, axis=0)

    return TierStats(
        features=features,
        medians=medians,
        stds=stds,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )


def _fit_pca(
    train_df: pd.DataFrame,
    features: list[str],
    n_components: int,
    random_state: int,
) -> tuple[list[str], SimpleImputer | None, StandardScaler | None, PCA | None]:
    if not features:
        return [], None, None, None
    n_components = min(int(n_components), len(features), len(train_df) - 1)
    if n_components <= 0:
        return [], None, None, None

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    matrix = imputer.fit_transform(train_df[features])
    matrix = scaler.fit_transform(matrix)
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(matrix)
    return features, imputer, scaler, pca


def _fit_rest_projection(
    train_df: pd.DataFrame,
    features: list[str],
    n_components: int,
    random_state: int,
    method: str,
):
    if not features:
        return [], None, None, None
    n_components = min(int(n_components), len(features), len(train_df) - 1)
    if n_components <= 0:
        return [], None, None, None

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    matrix = imputer.fit_transform(train_df[features])
    matrix = scaler.fit_transform(matrix)

    if method == "pca":
        projector = PCA(n_components=n_components, random_state=random_state)
    elif method == "random_projection":
        projector = GaussianRandomProjection(n_components=n_components, random_state=random_state)
    else:
        raise ValueError("method must be 'pca' or 'random_projection'.")

    projector.fit(matrix)
    return features, imputer, scaler, projector


def fit_selective_transform(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    feature_policy: dict[str, list[str]],
    experiment: dict,
    seed: int,
) -> SelectiveTransform:
    params = experiment["params"]
    method = experiment["method"]
    linkability_features = list(feature_policy["linkability_dominant"])
    shared_features = list(feature_policy["shared_high"])
    preserved_utility_features = list(feature_policy["utility_dominant"])

    linkability_noise_fraction = float(params.get("linkability_noise_fraction", 0.0))
    shared_noise_fraction = float(params.get("shared_noise_fraction", 0.0))
    winsor_lower = params.get("winsor_lower_quantile")
    winsor_upper = params.get("winsor_upper_quantile")

    pca_features: list[str] | None = None
    pca_imputer = None
    pca_scaler = None
    pca = None
    protected_rest_features = None
    rest_imputer = None
    rest_scaler = None
    rest_pca = None
    rest_projector = None
    output_feature_names = list(feature_columns)

    if method in {"selective_pca", "selective_pca_tiered"}:
        pca_features, pca_imputer, pca_scaler, pca = _fit_pca(
            train_df=train_df,
            features=linkability_features,
            n_components=int(params.get("pca_components", 10)),
            random_state=int(params.get("random_state", seed)),
        )
        if pca is not None:
            pca_names = [f"selective_link_pca_{idx:03d}" for idx in range(pca.n_components_)]
            output_feature_names = [feature for feature in feature_columns if feature not in set(pca_features)] + pca_names
            # PCA replaces the strongest linkability features, so we do not also add direct noise to them.
            linkability_noise_fraction = 0.0

    if method in {"preserve_utility_pca_rest", "preserve_utility_rp_rest"}:
        preserved_set = set(preserved_utility_features)
        rest_features = [feature for feature in feature_columns if feature not in preserved_set]
        projection_method = "pca" if method == "preserve_utility_pca_rest" else "random_projection"
        (
            protected_rest_features,
            rest_imputer,
            rest_scaler,
            rest_projector_obj,
        ) = _fit_rest_projection(
            train_df=train_df,
            features=rest_features,
            n_components=int(params.get("n_components", 50)),
            random_state=int(params.get("random_state", seed)),
            method=projection_method,
        )
        if projection_method == "pca":
            rest_pca = rest_projector_obj
            rest_prefix = "protected_rest_pca"
        else:
            rest_projector = rest_projector_obj
            rest_prefix = "protected_rest_rp"
        if rest_projector_obj is not None:
            rest_names = [f"{rest_prefix}_{idx:03d}" for idx in range(int(rest_projector_obj.n_components))]
            output_feature_names = preserved_utility_features + rest_names

    link_stats = _fit_tier_stats(
        train_df=train_df,
        features=linkability_features if linkability_noise_fraction > 0 else [],
        lower_quantile=winsor_lower,
        upper_quantile=winsor_upper,
    )
    shared_stats = _fit_tier_stats(
        train_df=train_df,
        features=shared_features if shared_noise_fraction > 0 else [],
        lower_quantile=None,
        upper_quantile=None,
    )

    return SelectiveTransform(
        name=experiment["name"],
        method=method,
        feature_columns=feature_columns,
        output_feature_names=output_feature_names,
        linkability_features=linkability_features,
        shared_features=shared_features,
        preserved_utility_features=preserved_utility_features,
        linkability_noise_fraction=linkability_noise_fraction,
        shared_noise_fraction=shared_noise_fraction,
        winsor_lower_quantile=float(winsor_lower) if winsor_lower is not None else None,
        winsor_upper_quantile=float(winsor_upper) if winsor_upper is not None else None,
        random_state=seed,
        linkability_tier_stats=link_stats,
        shared_tier_stats=shared_stats,
        pca_features=pca_features,
        pca_imputer=pca_imputer,
        pca_scaler=pca_scaler,
        pca=pca,
        protected_rest_features=protected_rest_features,
        rest_imputer=rest_imputer,
        rest_scaler=rest_scaler,
        rest_pca=rest_pca,
        rest_projector=rest_projector,
    )


def _copy_metadata(df: pd.DataFrame) -> pd.DataFrame:
    metadata_columns = [col for col in df.columns if col in METADATA_COLUMNS]
    return df[metadata_columns].copy()


def _apply_tier(
    feature_df: pd.DataFrame,
    tier_stats: TierStats | None,
    noise_fraction: float,
    random_state: int,
    split_name: str,
) -> pd.DataFrame:
    if tier_stats is None or not tier_stats.features:
        return feature_df

    transformed = feature_df.copy()
    values = transformed[tier_stats.features].apply(pd.to_numeric, errors="coerce").fillna(tier_stats.medians)
    if tier_stats.lower_bounds is not None and tier_stats.upper_bounds is not None:
        values = values.clip(lower=tier_stats.lower_bounds, upper=tier_stats.upper_bounds, axis=1)

    if noise_fraction > 0:
        offset = 0 if split_name == "train" else 10_000
        rng = np.random.default_rng(random_state + offset)
        noise = rng.normal(
            loc=0.0,
            scale=tier_stats.stds.to_numpy(dtype=float) * float(noise_fraction),
            size=values.shape,
        )
        values = values + noise

    transformed.loc[:, tier_stats.features] = values
    return transformed


def transform_selective_dataframe(df: pd.DataFrame, fitted: SelectiveTransform, split_name: str) -> pd.DataFrame:
    metadata_df = _copy_metadata(df)
    feature_df = df[fitted.feature_columns].copy().astype(float)

    feature_df = _apply_tier(
        feature_df=feature_df,
        tier_stats=fitted.linkability_tier_stats,
        noise_fraction=fitted.linkability_noise_fraction,
        random_state=fitted.random_state,
        split_name=split_name,
    )
    feature_df = _apply_tier(
        feature_df=feature_df,
        tier_stats=fitted.shared_tier_stats,
        noise_fraction=fitted.shared_noise_fraction,
        random_state=fitted.random_state + 1_000,
        split_name=split_name,
    )

    if fitted.pca is not None and fitted.pca_features:
        pca_matrix = fitted.pca_imputer.transform(df[fitted.pca_features])
        pca_matrix = fitted.pca_scaler.transform(pca_matrix)
        pca_matrix = fitted.pca.transform(pca_matrix)
        pca_names = [f"selective_link_pca_{idx:03d}" for idx in range(fitted.pca.n_components_)]
        pca_df = pd.DataFrame(pca_matrix, columns=pca_names, index=feature_df.index)
        feature_df = feature_df.drop(columns=fitted.pca_features)
        feature_df = pd.concat([feature_df, pca_df], axis=1)

    if fitted.protected_rest_features:
        rest_matrix = fitted.rest_imputer.transform(df[fitted.protected_rest_features])
        rest_matrix = fitted.rest_scaler.transform(rest_matrix)
        if fitted.rest_pca is not None:
            rest_matrix = fitted.rest_pca.transform(rest_matrix)
            rest_names = [f"protected_rest_pca_{idx:03d}" for idx in range(fitted.rest_pca.n_components_)]
        elif fitted.rest_projector is not None:
            rest_matrix = fitted.rest_projector.transform(rest_matrix)
            rest_names = [f"protected_rest_rp_{idx:03d}" for idx in range(int(fitted.rest_projector.n_components))]
        else:
            raise ValueError("Protected rest transform is missing fitted PCA/projector.")
        rest_df = pd.DataFrame(rest_matrix, columns=rest_names, index=feature_df.index)
        feature_df = feature_df.drop(columns=fitted.protected_rest_features)
        feature_df = pd.concat([feature_df, rest_df], axis=1)

    return pd.concat(
        [
            metadata_df.reset_index(drop=True),
            feature_df[fitted.output_feature_names].reset_index(drop=True),
        ],
        axis=1,
    )


def selective_transform_metadata(fitted: SelectiveTransform) -> dict:
    metadata = {
        "method": fitted.method,
        "n_input_features": len(fitted.feature_columns),
        "n_output_features": len(fitted.output_feature_names),
        "n_linkability_dominant_features": len(fitted.linkability_features),
        "n_shared_high_features": len(fitted.shared_features),
        "n_preserved_utility_dominant_features": len(fitted.preserved_utility_features),
        "linkability_features": fitted.linkability_features,
        "shared_features": fitted.shared_features,
        "preserved_utility_features": fitted.preserved_utility_features,
        "linkability_noise_fraction_of_train_std": fitted.linkability_noise_fraction,
        "shared_noise_fraction_of_train_std": fitted.shared_noise_fraction,
        "winsor_lower_quantile": fitted.winsor_lower_quantile,
        "winsor_upper_quantile": fitted.winsor_upper_quantile,
    }
    if fitted.pca is not None:
        metadata["pca_replaced_features"] = fitted.pca_features
        metadata["pca_n_components"] = int(fitted.pca.n_components_)
        metadata["pca_explained_variance_ratio_sum"] = float(fitted.pca.explained_variance_ratio_.sum())
    if fitted.rest_pca is not None:
        metadata["protected_rest_features"] = fitted.protected_rest_features
        metadata["protected_rest_transform"] = "pca"
        metadata["protected_rest_n_components"] = int(fitted.rest_pca.n_components_)
        metadata["protected_rest_explained_variance_ratio_sum"] = float(
            fitted.rest_pca.explained_variance_ratio_.sum()
        )
    if fitted.rest_projector is not None:
        metadata["protected_rest_features"] = fitted.protected_rest_features
        metadata["protected_rest_transform"] = "random_projection"
        metadata["protected_rest_n_components"] = int(fitted.rest_projector.n_components)
    return metadata


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
        "pca_explained_variance_ratio_sum",
        "protected_rest_explained_variance_ratio_sum",
    ]
    summary = metrics_df.groupby(
        [
            "transform_name",
            "method",
            "n_output_features",
            "n_linkability_dominant_features",
            "n_shared_high_features",
            "n_preserved_utility_dominant_features",
        ],
        as_index=False,
    )[metric_cols].agg(["mean", "std"])
    summary.columns = [
        "_".join([part for part in col if part]) if isinstance(col, tuple) else col for col in summary.columns
    ]
    return summary


def filter_experiments(experiments: list[dict], requested_names: list[str] | None) -> list[dict]:
    if not requested_names:
        return experiments
    requested = set(requested_names)
    available = {experiment["name"] for experiment in experiments}
    missing = sorted(requested - available)
    if missing:
        raise ValueError(f"Unknown experiment names: {missing}. Available: {sorted(available)}")
    return [experiment for experiment in experiments if experiment["name"] in requested]


def main() -> None:
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    max_chunks = None if args.max_chunks == 0 else args.max_chunks
    features_df, manifest, chunk_files = load_feature_dataset(args.dataset_dir, max_chunks=max_chunks)
    feature_columns = get_feature_columns(features_df)

    candidates_df = pd.read_csv(args.candidates_path)
    feature_policy, feature_policy_df = build_feature_policy(
        candidates_df=candidates_df,
        feature_columns=feature_columns,
        top_per_type=args.top_per_type,
    )

    metrics_rows = []
    transform_protocols = []

    for seed in args.seeds:
        train_df, test_df = split_segments_by_patient_stratified(
            features_df,
            group_col="patient_id",
            label_col="utility_label",
            test_size=args.test_size,
            random_state=seed,
        )

        experiments = filter_experiments(
            get_selective_transform_suite(seed, args.pca_components),
            requested_names=args.experiments,
        )
        for experiment in experiments:
            print(f"Seed {seed}: fitting {experiment['name']}...")
            fitted = fit_selective_transform(
                train_df=train_df,
                feature_columns=feature_columns,
                feature_policy=feature_policy,
                experiment=experiment,
                seed=seed,
            )
            train_transformed = transform_selective_dataframe(train_df, fitted, split_name="train")
            test_transformed = transform_selective_dataframe(test_df, fitted, split_name="test")

            utility_out = evaluate_utility_model_on_split(
                train_df=train_transformed,
                test_df=test_transformed,
                model_name=args.utility_model,
                target_col="utility_label",
            )
            utility_eval = utility_out["evaluation"]

            linkability_out = run_linkability_baselines_on_split(
                train_df=train_transformed,
                test_df=test_transformed,
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

            fitted_metadata = selective_transform_metadata(fitted)
            metrics_rows.append(
                {
                    "run_name": args.run_name,
                    "seed": seed,
                    "transform_name": fitted.name,
                    "method": fitted.method,
                    "params_json": json.dumps(experiment["params"], sort_keys=True),
                    "n_input_features": len(fitted.feature_columns),
                    "n_output_features": len(fitted.output_feature_names),
                    "n_linkability_dominant_features": len(fitted.linkability_features),
                    "n_shared_high_features": len(fitted.shared_features),
                    "n_preserved_utility_dominant_features": len(fitted.preserved_utility_features),
                    "train_segments": len(train_df),
                    "test_segments": len(test_df),
                    "train_patients": train_df["patient_id"].nunique(),
                    "test_patients": test_df["patient_id"].nunique(),
                    "utility_model": utility_eval["model"],
                    "utility_f1": utility_eval["f1_score"],
                    "utility_balanced_accuracy": utility_eval["balanced_accuracy"],
                    "utility_roc_auc": utility_eval["roc_auc"],
                    "utility_pr_auc": utility_eval["pr_auc"],
                    "linkability_model": xgb_row["model"],
                    "linkability_f1": xgb_row["f1_score"],
                    "linkability_balanced_accuracy": xgb_row["balanced_accuracy"],
                    "linkability_roc_auc": xgb_row["roc_auc"],
                    "linkability_pr_auc": xgb_row["pr_auc"],
                    "link_train_pairs": len(linkability_out["train_pair_df"]),
                    "link_test_pairs": len(linkability_out["test_pair_df"]),
                    "link_train_positive_prevalence": float(linkability_out["train_pair_df"]["pair_label"].mean()),
                    "link_test_positive_prevalence": float(linkability_out["test_pair_df"]["pair_label"].mean()),
                    "pca_explained_variance_ratio_sum": fitted_metadata.get(
                        "pca_explained_variance_ratio_sum"
                    ),
                    "protected_rest_explained_variance_ratio_sum": fitted_metadata.get(
                        "protected_rest_explained_variance_ratio_sum"
                    ),
                }
            )

            transform_protocols.append(
                {
                    "seed": seed,
                    "transform_name": fitted.name,
                    "requested_method": experiment["method"],
                    "requested_params": experiment["params"],
                    "fitted_transform": fitted_metadata,
                }
            )

    metrics_df = pd.DataFrame(metrics_rows)
    summary_df = summarize_metrics(metrics_df)

    metrics_path = args.output_dir / f"{args.run_name}_metrics.csv"
    summary_path = args.output_dir / f"{args.run_name}_summary.csv"
    feature_policy_path = args.output_dir / f"{args.run_name}_feature_policy.csv"
    protocol_path = args.output_dir / f"{args.run_name}_protocol.json"

    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    feature_policy_df.to_csv(feature_policy_path, index=False)

    protocol = {
        "run_name": args.run_name,
        "created_at_epoch": started,
        "elapsed_seconds": round(time.time() - started, 2),
        "dataset_dir": str(args.dataset_dir),
        "candidates_path": str(args.candidates_path),
        "feature_policy_source": (
            "precomputed full-dataset model-based utility/linkability importance; exploratory, not nested within each split"
        ),
        "manifest_total_segments": manifest.get("total_segments"),
        "manifest_window_sec": manifest.get("window_sec"),
        "manifest_step_sec": manifest.get("step_sec"),
        "loaded_chunks": len(chunk_files),
        "max_chunks": max_chunks,
        "loaded_segments": len(features_df),
        "test_size": args.test_size,
        "seeds": args.seeds,
        "top_per_type": args.top_per_type,
        "feature_policy": feature_policy,
        "utility_protocol": {
            "split": "patient-level stratified split",
            "target": "utility_label",
            "model": args.utility_model,
        },
        "transformation_protocol": {
            "fit_unit": "train split segment-level feature matrix",
            "selection_unit": "feature groups from prior feature-importance overlap analysis",
            "application": "fitted on train, applied separately to train and test",
            "noise_scale": "fraction of each selected feature's train-split standard deviation, applied in original feature units",
            "pca_scale": "median imputation and standard scaling fitted on train for selected linkability-dominant features only",
            "pair_order": "transform features first; build linkability pairs after transformation",
            "preserve_rule": "utility_dominant features are excluded from selective transformation",
        },
        "linkability_protocol": {
            "representation": "absdiff",
            "model": "XGBoost",
            "max_pairs": args.link_max_pairs,
            "min_segment_gap": args.link_min_segment_gap,
            "max_positive_pairs_per_patient": args.link_max_positive_pairs_per_patient,
            "negative_strategy": args.negative_strategy,
            "hard_negative_pool_size": args.hard_negative_pool_size,
            "pair_prevalence": "balanced by construction",
            "test_pair_random_state": "seed + 1 inside pair builder",
        },
        "transform_protocols": transform_protocols,
        "metrics_path": str(metrics_path),
        "summary_path": str(summary_path),
        "feature_policy_path": str(feature_policy_path),
    }
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    print("Saved metrics:", metrics_path)
    print("Saved summary:", summary_path)
    print("Saved feature policy:", feature_policy_path)
    print("Saved protocol:", protocol_path)


if __name__ == "__main__":
    main()
