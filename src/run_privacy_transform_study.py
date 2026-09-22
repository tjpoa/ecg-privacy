from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

try:
    import joblib
except ImportError:  # pragma: no cover - optional persistence dependency
    joblib = None

try:
    from config import FINAL_SEGMENT_FEATURES_DIR, OUTPUTS_MODELS_DIR, OUTPUTS_TABLES_DIR
    from modeling import (
        evaluate_utility_model_on_split,
        get_feature_columns,
        run_linkability_baselines_on_split,
        split_segments_by_patient_stratified,
    )
    from privacy_transforms import fit_transform_feature_split
except ImportError:  # pragma: no cover - package import fallback
    from .config import FINAL_SEGMENT_FEATURES_DIR, OUTPUTS_MODELS_DIR, OUTPUTS_TABLES_DIR
    from .modeling import (
        evaluate_utility_model_on_split,
        get_feature_columns,
        run_linkability_baselines_on_split,
        split_segments_by_patient_stratified,
    )
    from .privacy_transforms import fit_transform_feature_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run split-aware privacy transformations for utility/linkability trade-off evaluation."
    )
    parser.add_argument("--dataset-dir", type=Path, default=FINAL_SEGMENT_FEATURES_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_TABLES_DIR)
    parser.add_argument("--models-dir", type=Path, default=OUTPUTS_MODELS_DIR / "privacy_transform_study")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--suite", choices=["focused", "full", "dp"], default="focused")
    parser.add_argument("--utility-model", default="LogisticRegression")
    parser.add_argument("--link-max-pairs", type=int, default=2000)
    parser.add_argument("--link-min-segment-gap", type=int, default=4)
    parser.add_argument("--link-max-positive-pairs-per-patient", type=int, default=3)
    parser.add_argument("--negative-strategy", choices=["random", "hard"], default="random")
    parser.add_argument("--hard-negative-pool-size", type=int, default=5)
    parser.add_argument("--save-fitted", action="store_true")
    return parser.parse_args()


def get_transform_suite(name: str, seed: int) -> list[dict]:
    focused = [
        {"name": "identity", "method": "identity", "params": {}},
        {"name": "pca_120", "method": "pca", "params": {"n_components": 120, "random_state": seed}},
    ]
    if name == "focused":
        return focused
    if name == "dp":
        return [
            {"name": "identity", "method": "identity", "params": {}},
            {"name": "noise_010", "method": "noise", "params": {"noise_std": 0.10, "random_state": seed}},
            {
                "name": "dp_gaussian_eps200_clip8",
                "method": "dp_gaussian_l2",
                "params": {"epsilon": 200.0, "delta": 1e-5, "clip_norm": 8.0, "random_state": seed},
            },
            {
                "name": "dp_gaussian_eps100_clip8",
                "method": "dp_gaussian_l2",
                "params": {"epsilon": 100.0, "delta": 1e-5, "clip_norm": 8.0, "random_state": seed},
            },
            {
                "name": "dp_gaussian_eps50_clip8",
                "method": "dp_gaussian_l2",
                "params": {"epsilon": 50.0, "delta": 1e-5, "clip_norm": 8.0, "random_state": seed},
            },
            {
                "name": "dp_gaussian_eps100_clip4",
                "method": "dp_gaussian_l2",
                "params": {"epsilon": 100.0, "delta": 1e-5, "clip_norm": 4.0, "random_state": seed},
            },
            {
                "name": "dp_gaussian_eps20_clip2",
                "method": "dp_gaussian_l2",
                "params": {"epsilon": 20.0, "delta": 1e-5, "clip_norm": 2.0, "random_state": seed},
            },
            {
                "name": "dp_gaussian_eps10_clip2",
                "method": "dp_gaussian_l2",
                "params": {"epsilon": 10.0, "delta": 1e-5, "clip_norm": 2.0, "random_state": seed},
            },
            {
                "name": "dp_gaussian_eps5_clip2",
                "method": "dp_gaussian_l2",
                "params": {"epsilon": 5.0, "delta": 1e-5, "clip_norm": 2.0, "random_state": seed},
            },
            {
                "name": "dp_gaussian_eps10_clip4",
                "method": "dp_gaussian_l2",
                "params": {"epsilon": 10.0, "delta": 1e-5, "clip_norm": 4.0, "random_state": seed},
            },
            {
                "name": "dp_laplace_eps20_clip2",
                "method": "dp_laplace_l1",
                "params": {"epsilon": 20.0, "clip_norm": 2.0, "random_state": seed},
            },
        ]

    return focused + [
        {"name": "standard", "method": "standard", "params": {}},
        {"name": "robust", "method": "robust", "params": {}},
        {"name": "minmax", "method": "minmax", "params": {}},
        {"name": "quant_dec2", "method": "quantization", "params": {"decimals": 2}},
        {"name": "quant_dec1", "method": "quantization", "params": {"decimals": 1}},
        {"name": "quant_dec0", "method": "quantization", "params": {"decimals": 0}},
        {"name": "noise_001", "method": "noise", "params": {"noise_std": 0.01, "random_state": seed}},
        {"name": "noise_005", "method": "noise", "params": {"noise_std": 0.05, "random_state": seed}},
        {"name": "noise_010", "method": "noise", "params": {"noise_std": 0.10, "random_state": seed}},
        {
            "name": "winsor_01_99",
            "method": "winsorization",
            "params": {"lower_quantile": 0.01, "upper_quantile": 0.99},
        },
        {
            "name": "winsor_05_95",
            "method": "winsorization",
            "params": {"lower_quantile": 0.05, "upper_quantile": 0.95},
        },
        {
            "name": "winsor_10_90",
            "method": "winsorization",
            "params": {"lower_quantile": 0.10, "upper_quantile": 0.90},
        },
        {"name": "rp_120", "method": "random_projection", "params": {"n_components": 120, "random_state": seed}},
    ]


def load_feature_dataset(dataset_dir: Path, max_chunks: int | None) -> tuple[pd.DataFrame, dict, list[Path]]:
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunk_files = [dataset_dir / item["chunk_file"] for item in manifest["chunks"]]
    if max_chunks is not None:
        chunk_files = chunk_files[:max_chunks]

    features_df = pd.concat(
        [pd.read_csv(chunk_file, compression="gzip", low_memory=False) for chunk_file in chunk_files],
        ignore_index=True,
    )
    return features_df, manifest, chunk_files


def fitted_transform_metadata(fitted) -> dict:
    metadata = {
        "method": fitted.name,
        "n_input_features": len(fitted.feature_columns),
        "n_output_features": len(fitted.output_feature_names),
        "input_features": fitted.feature_columns,
        "output_features": fitted.output_feature_names,
        "imputer": "SimpleImputer(strategy='median')" if fitted.imputer is not None else None,
    }
    if fitted.scaler is not None:
        metadata["scaler"] = fitted.scaler.__class__.__name__
    if fitted.pca is not None:
        metadata["pca_n_components"] = int(fitted.pca.n_components_)
        metadata["pca_explained_variance_ratio_sum"] = float(fitted.pca.explained_variance_ratio_.sum())
    if fitted.projector is not None:
        metadata["projector"] = fitted.projector.__class__.__name__
        metadata["projector_n_components"] = int(fitted.projector.n_components)
    if fitted.decimals is not None:
        metadata["quantization_decimals"] = int(fitted.decimals)
    if fitted.noise_std is not None:
        metadata["noise_distribution"] = "gaussian"
        metadata["noise_std"] = float(fitted.noise_std)
        metadata["noise_random_state"] = int(fitted.random_state)
        metadata["test_noise_random_state_offset"] = 10_000
    if fitted.dp_mechanism is not None:
        metadata["dp_mechanism"] = fitted.dp_mechanism
        metadata["dp_epsilon"] = float(fitted.dp_epsilon)
        metadata["dp_delta"] = float(fitted.dp_delta)
        metadata["dp_clip_norm"] = float(fitted.dp_clip_norm)
        metadata["dp_noise_scale"] = float(fitted.dp_noise_scale)
        metadata["dp_random_state"] = int(fitted.random_state)
        metadata["dp_sensitivity"] = float(2.0 * fitted.dp_clip_norm)
        metadata["dp_scope_note"] = (
            "Noise is calibrated for releasing each clipped standardized feature vector. "
            "End-to-end DP requires public or DP-fitted preprocessing parameters."
        )
    if fitted.lower_bounds is not None and fitted.upper_bounds is not None:
        metadata["winsorization_fitted_on"] = "train feature matrix after median imputation"
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
    ]
    summary = metrics_df.groupby(["transform_name", "method", "n_output_features"], as_index=False)[metric_cols].agg(
        ["mean", "std"]
    )
    summary.columns = [
        "_".join([part for part in col if part]) if isinstance(col, tuple) else col for col in summary.columns
    ]
    return summary


def main() -> None:
    args = parse_args()
    started = time.time()
    run_name = args.run_name or f"privacy_transform_{args.suite}_{int(started)}"
    output_dir = args.output_dir
    models_dir = args.models_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_fitted:
        if joblib is None:
            raise ImportError("joblib is required when --save-fitted is used.")
        models_dir.mkdir(parents=True, exist_ok=True)

    features_df, manifest, chunk_files = load_feature_dataset(args.dataset_dir, args.max_chunks)
    feature_columns = get_feature_columns(features_df)

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

        for experiment in get_transform_suite(args.suite, seed):
            transform_name = experiment["name"]
            method = experiment["method"]
            params = experiment["params"]
            print(f"Seed {seed}: fitting {transform_name}...")

            train_transformed, test_transformed, fitted = fit_transform_feature_split(
                train_df=train_df,
                test_df=test_df,
                method=method,
                feature_columns=feature_columns,
                **params,
            )

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

            fitted_path = None
            if args.save_fitted:
                fitted_path = models_dir / f"{transform_name}_seed{seed}.joblib"
                joblib.dump(fitted, fitted_path)

            metrics_rows.append(
                {
                    "run_name": run_name,
                    "seed": seed,
                    "transform_name": transform_name,
                    "method": method,
                    "params_json": json.dumps(params, sort_keys=True),
                    "n_input_features": len(fitted.feature_columns),
                    "n_output_features": len(fitted.output_feature_names),
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
                    "fitted_transform_path": str(fitted_path) if fitted_path is not None else None,
                }
            )

            transform_protocols.append(
                {
                    "seed": seed,
                    "transform_name": transform_name,
                    "requested_method": method,
                    "requested_params": params,
                    "fitted_transform": fitted_transform_metadata(fitted),
                    "fitted_transform_path": str(fitted_path) if fitted_path is not None else None,
                }
            )

    metrics_df = pd.DataFrame(metrics_rows)
    summary_df = summarize_metrics(metrics_df)

    metrics_path = output_dir / f"{run_name}_metrics.csv"
    summary_path = output_dir / f"{run_name}_summary.csv"
    protocol_path = output_dir / f"{run_name}_protocol.json"

    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    protocol = {
        "run_name": run_name,
        "created_at_epoch": started,
        "elapsed_seconds": round(time.time() - started, 2),
        "dataset_dir": str(args.dataset_dir),
        "manifest_window_sec": manifest.get("window_sec"),
        "manifest_step_sec": manifest.get("step_sec"),
        "manifest_total_segments": manifest.get("total_segments"),
        "loaded_chunks": len(chunk_files),
        "max_chunks": args.max_chunks,
        "loaded_segments": len(features_df),
        "test_size": args.test_size,
        "seeds": args.seeds,
        "suite": args.suite,
        "utility_protocol": {
            "split": "patient-level stratified split",
            "target": "utility_label",
            "model": args.utility_model,
        },
        "transformation_protocol": {
            "fit_unit": "train split segment-level feature matrix",
            "imputation": "median imputation fitted on train before adjustable transforms",
            "application": "fitted on train, applied separately to train and test",
            "pair_order": "transform features first; build linkability pairs after transformation",
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
            "test_pair_random_state": "seed + 1",
        },
        "transform_protocols": transform_protocols,
        "metrics_path": str(metrics_path),
        "summary_path": str(summary_path),
    }
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    print("Saved metrics:", metrics_path)
    print("Saved summary:", summary_path)
    print("Saved protocol:", protocol_path)
    if args.save_fitted:
        print("Saved fitted transforms under:", models_dir)


if __name__ == "__main__":
    main()
