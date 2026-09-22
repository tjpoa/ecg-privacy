from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

try:
    from config import OUTPUTS_TABLES_DIR
except ImportError:  # pragma: no cover - package import fallback
    from .config import OUTPUTS_TABLES_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create concise article tables and paired bootstrap effects for the privacy attack extensions."
    )
    parser.add_argument(
        "--attribute-prefix",
        type=Path,
        default=OUTPUTS_TABLES_DIR / "attribute_inference_full_multiseed",
    )
    parser.add_argument(
        "--reconstruction-prefix",
        type=Path,
        default=OUTPUTS_TABLES_DIR / "reconstruction_full_multiseed",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_TABLES_DIR)
    parser.add_argument("--run-name", default="privacy_attack_extensions")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260714)
    return parser.parse_args()


def metric_effect(task: str, y_true: np.ndarray, baseline_pred: np.ndarray, comparison_pred: np.ndarray) -> float:
    if task in {"sex", "age_group"}:
        positive = y_true == 1
        negative = y_true == 0

        def balanced_accuracy(prediction: np.ndarray) -> float:
            sensitivity = float(np.mean(prediction[positive] == 1)) if positive.any() else np.nan
            specificity = float(np.mean(prediction[negative] == 0)) if negative.any() else np.nan
            return 0.5 * (sensitivity + specificity)

        return float(balanced_accuracy(comparison_pred) - balanced_accuracy(baseline_pred))
    return float(
        mean_absolute_error(y_true, baseline_pred)
        - mean_absolute_error(y_true, comparison_pred)
    )


def paired_seed_data(
    predictions_df: pd.DataFrame,
    seed: int,
    task: str,
    representation: str,
    comparison_condition: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = predictions_df[
        (predictions_df["seed"] == seed)
        & (predictions_df["task"] == task)
        & (predictions_df["representation"] == representation)
        & predictions_df["test_condition"].isin(["single_segment", comparison_condition])
    ]
    baseline = selected[selected["test_condition"] == "single_segment"][
        ["patient_id", "y_true", "y_pred"]
    ].rename(columns={"y_pred": "baseline_pred", "y_true": "baseline_true"})
    comparison = selected[selected["test_condition"] == comparison_condition][
        ["patient_id", "y_true", "y_pred"]
    ].rename(columns={"y_pred": "comparison_pred", "y_true": "comparison_true"})
    paired = baseline.merge(comparison, on="patient_id", how="inner")
    if not np.allclose(paired["baseline_true"], paired["comparison_true"]):
        raise AssertionError("Paired attribute targets differ between conditions.")
    return (
        paired["baseline_true"].to_numpy(),
        paired["baseline_pred"].to_numpy(),
        paired["comparison_pred"].to_numpy(),
    )


def stratified_paired_bootstrap(
    predictions_df: pd.DataFrame,
    task: str,
    representation: str,
    comparison_condition: str,
    replicates: int,
    seed: int,
) -> dict:
    seed_values = sorted(predictions_df["seed"].unique())
    arrays = [
        paired_seed_data(
            predictions_df,
            seed=int(seed_value),
            task=task,
            representation=representation,
            comparison_condition=comparison_condition,
        )
        for seed_value in seed_values
    ]
    point_effects = [metric_effect(task, *values) for values in arrays]
    rng = np.random.default_rng(seed)
    bootstrap_effects = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        replicate_effects = []
        for y_true, baseline_pred, comparison_pred in arrays:
            indices = rng.integers(0, len(y_true), size=len(y_true))
            replicate_effects.append(
                metric_effect(
                    task,
                    y_true[indices],
                    baseline_pred[indices],
                    comparison_pred[indices],
                )
            )
        bootstrap_effects[replicate] = np.mean(replicate_effects)
    metric_name = "balanced_accuracy_gain" if task in {"sex", "age_group"} else "mae_reduction_years"
    return {
        "task": task,
        "representation": representation,
        "comparison_condition": comparison_condition,
        "effect_metric": metric_name,
        "effect_mean_across_seeds": float(np.mean(point_effects)),
        "effect_seed_std": float(np.std(point_effects, ddof=1)),
        "bootstrap_ci_low": float(np.quantile(bootstrap_effects, 0.025)),
        "bootstrap_ci_high": float(np.quantile(bootstrap_effects, 0.975)),
        "bootstrap_replicates": int(replicates),
        "seeds": int(len(seed_values)),
        "paired_queries_per_seed_min": int(min(len(values[0]) for values in arrays)),
    }


def build_attribute_article_table(metrics_df: pd.DataFrame, link_df: pd.DataFrame) -> pd.DataFrame:
    representations = ["raw_features", "encoder", "encoder_gaussian_noise"]
    rows = []
    for representation in representations:
        row = {"representation": representation}
        link_values = link_df[link_df["representation"] == representation]["selection_precision"]
        row["link_selection_precision_mean"] = link_values.mean()
        row["link_selection_precision_std"] = link_values.std(ddof=1)
        for task, metric, prefix in [
            ("sex", "balanced_accuracy", "sex_ba"),
            ("age_group", "balanced_accuracy", "age65_ba"),
            ("age", "mae_years", "age_mae"),
        ]:
            selected = metrics_df[
                (metrics_df["representation"] == representation) & (metrics_df["task"] == task)
            ]
            for condition in ["single_segment", "oracle_group", "link_selected_group", "random_group"]:
                values = selected[selected["test_condition"] == condition][metric]
                row[f"{prefix}_{condition}_mean"] = values.mean()
                row[f"{prefix}_{condition}_std"] = values.std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def build_reconstruction_article_table(metrics_df: pd.DataFrame, pca_df: pd.DataFrame) -> pd.DataFrame:
    conditions = [
        ("waveform_baseline", "train_mean"),
        ("waveform_baseline", "target_pca_ceiling"),
        ("raw_features", "ridge"),
        ("raw_features", "mlp"),
        ("encoder", "ridge"),
        ("encoder", "mlp"),
        ("encoder_gaussian_noise", "ridge"),
        ("encoder_gaussian_noise", "mlp"),
    ]
    rows = []
    for representation, decoder in conditions:
        selected = metrics_df[
            (metrics_df["representation"] == representation) & (metrics_df["decoder"] == decoder)
        ]
        row = {"representation": representation, "decoder": decoder}
        for metric in [
            "prd_mean",
            "snr_db_mean",
            "lead_ii_correlation_mean",
            "lead_ii_maxlag_correlation_mean",
            "lead_ii_spectral_correlation_mean",
        ]:
            row[f"{metric}_mean"] = selected[metric].mean()
            row[f"{metric}_std"] = selected[metric].std(ddof=1)
        rows.append(row)
    table = pd.DataFrame(rows)
    table.attrs["target_pca_explained_variance_mean"] = float(
        pca_df["explained_variance_ratio_sum"].mean()
    )
    return table


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    attribute_metrics = pd.read_csv(args.attribute_prefix.with_name(args.attribute_prefix.name + "_metrics.csv"))
    attribute_predictions = pd.read_csv(
        args.attribute_prefix.with_name(args.attribute_prefix.name + "_predictions.csv.gz"),
        compression="gzip",
    )
    link_metrics = pd.read_csv(
        args.attribute_prefix.with_name(args.attribute_prefix.name + "_link_selection.csv")
    )
    reconstruction_metrics_df = pd.read_csv(
        args.reconstruction_prefix.with_name(args.reconstruction_prefix.name + "_metrics.csv")
    )
    reconstruction_pca_df = pd.read_csv(
        args.reconstruction_prefix.with_name(args.reconstruction_prefix.name + "_target_pca.csv")
    )

    bootstrap_rows = []
    counter = 0
    for task in ["sex", "age_group", "age"]:
        for representation in ["raw_features", "encoder", "encoder_gaussian_noise"]:
            for condition in ["oracle_group", "link_selected_group", "random_group"]:
                bootstrap_rows.append(
                    stratified_paired_bootstrap(
                        predictions_df=attribute_predictions,
                        task=task,
                        representation=representation,
                        comparison_condition=condition,
                        replicates=args.bootstrap_replicates,
                        seed=args.bootstrap_seed + counter,
                    )
                )
                counter += 1

    bootstrap_df = pd.DataFrame(bootstrap_rows)
    attribute_table = build_attribute_article_table(attribute_metrics, link_metrics)
    reconstruction_table = build_reconstruction_article_table(
        reconstruction_metrics_df, reconstruction_pca_df
    )

    bootstrap_path = args.output_dir / f"{args.run_name}_attribute_bootstrap_effects.csv"
    attribute_path = args.output_dir / f"{args.run_name}_attribute_article_table.csv"
    reconstruction_path = args.output_dir / f"{args.run_name}_reconstruction_article_table.csv"
    bootstrap_df.to_csv(bootstrap_path, index=False)
    attribute_table.to_csv(attribute_path, index=False)
    reconstruction_table.to_csv(reconstruction_path, index=False)
    print("Saved bootstrap effects:", bootstrap_path)
    print("Saved attribute article table:", attribute_path)
    print("Saved reconstruction article table:", reconstruction_path)


if __name__ == "__main__":
    main()
