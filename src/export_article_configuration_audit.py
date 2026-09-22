"""Export the configuration-selection evidence and fixed article settings.

This script consolidates already-completed development experiments.  It does
not refit models or select configurations from the final article table.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"


def load_required(name: str) -> pd.DataFrame:
    path = TABLES / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def build_selection_evidence() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    baselines = load_required("final_baseline_summary.csv")
    for task, metric in [("utility", "balanced_accuracy"), ("linkability", "roc_auc")]:
        selected = baselines[baselines["task"].eq(task)].copy()
        if set(selected["model"]) != {"LogisticRegression", "XGBoost"}:
            raise ValueError(f"Unexpected {task} development candidates")
        for row in selected.itertuples(index=False):
            rows.append(
                {
                    "screen": f"{task}_model_family",
                    "candidate": row.model,
                    "development_seeds": 1,
                    "primary_metric": metric,
                    "primary_metric_mean": float(getattr(row, metric)),
                    "secondary_metric": "",
                    "secondary_metric_mean": "",
                    "decision": "retained" if row.status == "selected_baseline" else "comparison",
                    "interpretation": (
                        "fixed utility linear probe"
                        if task == "utility" and row.status == "selected_baseline"
                        else "fixed nonlinear linkability attacker"
                        if task == "linkability" and row.status == "selected_baseline"
                        else "development comparator"
                    ),
                }
            )

    encoder = load_required("supervised_bottleneck_encoder_seeds123456_metrics.csv")
    encoder_summary = (
        encoder.groupby("bottleneck_dim", as_index=False)
        .agg(
            development_seeds=("seed", "nunique"),
            utility_balanced_accuracy_mean=("utility_balanced_accuracy", "mean"),
            linkability_roc_auc_mean=("linkability_roc_auc", "mean"),
        )
        .sort_values("bottleneck_dim")
    )
    if encoder_summary["bottleneck_dim"].tolist() != [8, 16, 32]:
        raise ValueError("Unexpected encoder dimension screen")
    for row in encoder_summary.itertuples(index=False):
        rows.append(
            {
                "screen": "encoder_dimension",
                "candidate": f"{int(row.bottleneck_dim)}-D",
                "development_seeds": int(row.development_seeds),
                "primary_metric": "utility_balanced_accuracy",
                "primary_metric_mean": float(row.utility_balanced_accuracy_mean),
                "secondary_metric": "linkability_roc_auc",
                "secondary_metric_mean": float(row.linkability_roc_auc_mean),
                "decision": "retained" if int(row.bottleneck_dim) == 8 else "comparison",
                "interpretation": (
                    "smallest screened bottleneck with near-maximum utility and weakest linkability"
                    if int(row.bottleneck_dim) == 8
                    else "development comparator"
                ),
            }
        )

    operational = load_required("encoder_dp_operational_dim8_multiseed_summary.csv")
    operational = operational[operational["gallery_negatives"].eq(999)].copy()
    expected = {
        "encoder_identity",
        "encoder_dp_gaussian_eps100_clip2",
        "encoder_dp_gaussian_eps50_clip2",
        "encoder_dp_gaussian_eps50_clip4",
        "encoder_dp_gaussian_eps20_clip4",
    }
    if set(operational["transform_name"]) != expected:
        raise ValueError("Unexpected Gaussian development screen")
    decisions = {
        "encoder_identity": ("reference", "unperturbed encoder reference"),
        "encoder_dp_gaussian_eps100_clip2": ("comparison", "weaker-noise development comparator"),
        "encoder_dp_gaussian_eps50_clip2": ("retained", "high-utility operating point"),
        "encoder_dp_gaussian_eps50_clip4": ("retained", "stronger-perturbation operating point"),
        "encoder_dp_gaussian_eps20_clip4": ("comparison", "utility-degrading development boundary"),
    }
    for row in operational.sort_values("transform_name").itertuples(index=False):
        decision, interpretation = decisions[row.transform_name]
        rows.append(
            {
                "screen": "gaussian_operating_point",
                "candidate": row.transform_name,
                "development_seeds": 3,
                "primary_metric": "utility_balanced_accuracy",
                "primary_metric_mean": float(row.utility_balanced_accuracy_mean),
                "secondary_metric": "pairwise_linkability_roc_auc",
                "secondary_metric_mean": float(row.pairwise_linkability_roc_auc_mean),
                "decision": decision,
                "interpretation": interpretation,
            }
        )

    return pd.DataFrame(rows)


def build_fixed_configurations() -> pd.DataFrame:
    rows = [
        {
            "component": "utility_probe",
            "model": "LogisticRegression",
            "fixed_configuration": "training-median imputation; training-fitted StandardScaler; penalty=L2; C=1; solver=lbfgs; class_weight=balanced; max_iter=2000; random_state=42",
            "role": "fixed linear probe of accessible utility information",
        },
        {
            "component": "pairwise_and_gallery_linkability",
            "model": "XGBoost",
            "fixed_configuration": "absolute representation differences; binary-logistic XGBClassifier; 200 trees; max_depth=4; learning_rate=0.05; subsample=0.8; colsample_bytree=0.8; eval_metric=logloss; random_state=42; 2000 positive and 2000 negative pairs per split; minimum segment gap=4; at most 3 positive pairs per record",
            "role": "fixed nonlinear attacker on absolute representation differences",
        },
        {
            "component": "linear_compression_controls",
            "model": "PCA and Gaussian random projection",
            "fixed_configuration": "PCA(whiten=False) and GaussianRandomProjection; 8-D in the main matched comparison; 64-D sensitivity; transformations fitted on training rows only; split seed used as random_state",
            "role": "data-dependent and data-independent linear controls",
        },
        {
            "component": "supervised_encoder",
            "model": "MLPClassifier",
            "fixed_configuration": "training-median imputation and StandardScaler; MLPClassifier hidden layers=(128,8), with the final hidden activation released; ReLU; Adam; alpha=1e-4; batch=1024; learning_rate_init=1e-3; max_iter=25; early_stopping=True; validation_fraction=0.1; n_iter_no_change=8; split seed as random_state; at most 40000 class-balanced training segments",
            "role": "utility-supervised representation generator with a fixed computational budget",
        },
        {
            "component": "attribute_attackers",
            "model": "XGBoost classifier/regressor",
            "fixed_configuration": "training-median imputation; 200 trees; max_depth=4; learning_rate=0.05; subsample=0.8; colsample_bytree=0.8; tree_method=hist; n_jobs=4; split seed as random_state; classifiers use objective=binary:logistic, eval_metric=logloss, and training class ratio; age regressor uses objective=reg:squarederror and eval_metric=rmse",
            "role": "fixed-capacity tabular attribute attackers",
        },
        {
            "component": "linear_reconstruction_attacker",
            "model": "Ridge",
            "fixed_configuration": "training-median imputation and StandardScaler on released inputs; training-fitted StandardScaler on 128 target-PCA scores; Ridge alpha=1, fit_intercept=True, solver=auto",
            "role": "linear inversion baseline",
        },
        {
            "component": "nonlinear_reconstruction_attacker",
            "model": "MLPRegressor",
            "fixed_configuration": "same input and target-score scaling as Ridge; MLPRegressor hidden layers=(256,128); squared-error loss; ReLU; Adam; alpha=1e-4; batch=256; learning_rate_init=1e-3; max_iter=100; early_stopping=True; validation_fraction=0.1; n_iter_no_change=10; split seed as random_state",
            "role": "nonlinear inversion attacker",
        },
        {
            "component": "waveform_target_compression",
            "model": "PCA",
            "fixed_configuration": "PCA with 128 components; randomized SVD; iterated_power=3; split seed as random_state; fitted only on training waveforms",
            "role": "computational approximation, audited through its direct-reconstruction ceiling",
        },
        {
            "component": "gaussian_perturbation",
            "model": "L2 clipping plus isotropic Gaussian noise",
            "fixed_configuration": "standardize embeddings; clip to C in {2,4}; add isotropic Gaussian noise with sigma in {0.3876,0.7752}; noise multiplier=sigma/C=0.1938; no nominal epsilon/delta parametrization",
            "role": "two representative operating points, not an optimized or end-to-end DP mechanism",
        },
        {
            "component": "software",
            "model": "scikit-learn and XGBoost",
            "fixed_configuration": "scikit-learn=1.7.2; XGBoost=3.2.0; NumPy=2.2.6; pandas=2.3.3; SciPy=1.15.3",
            "role": "versions used to reproduce the reported estimators and metrics",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    selection = build_selection_evidence()
    configurations = build_fixed_configurations()

    selection_path = TABLES / "article_configuration_selection_evidence.csv"
    configuration_path = TABLES / "article_model_configuration_audit.csv"
    protocol_path = TABLES / "article_configuration_selection_protocol.json"
    selection.to_csv(selection_path, index=False)
    configurations.to_csv(configuration_path, index=False)

    protocol = {
        "purpose": "Consolidate development-stage configuration evidence used to justify fixed article settings.",
        "selection_scope": (
            "Development screens support representative study-design choices; they are not a nested, "
            "independent hyperparameter-optimization study and do not establish global optimality."
        ),
        "final_evaluation": (
            "The controlled article comparison fixes model families, settings, sample budgets, and pair budgets "
            "before reporting the final three-seed means."
        ),
        "software_versions": {
            "scikit_learn": "1.7.2",
            "xgboost": "3.2.0",
            "numpy": "2.2.6",
            "pandas": "2.3.3",
            "scipy": "1.15.3",
        },
        "gaussian_reporting_policy": {
            "decision": "remove nominal epsilon/delta parametrization from the article",
            "reported_parameters": ["noise_multiplier_m", "clip_norm_C", "noise_standard_deviation_sigma"],
            "retained_operating_points": [
                {"noise_multiplier_m": 0.1938, "clip_norm_C": 2.0, "noise_standard_deviation_sigma": 0.3876},
                {"noise_multiplier_m": 0.1938, "clip_norm_C": 4.0, "noise_standard_deviation_sigma": 0.7752},
            ],
            "rationale": (
                "These direct parameters fully reproduce the empirical release. No adjacency relation, validated "
                "sensitivity argument, or composition accountant supports an epsilon/delta privacy guarantee."
            ),
            "provenance_note": (
                "Historical raw run protocols retain the original nominal epsilon/delta generation metadata; "
                "those fields are provenance only and are not article-level privacy parameters."
            ),
        },
        "source_tables": [
            "final_baseline_summary.csv",
            "supervised_bottleneck_encoder_seeds123456_metrics.csv",
            "encoder_dp_operational_dim8_multiseed_summary.csv",
        ],
        "outputs": [selection_path.name, configuration_path.name],
    }
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"Saved {selection_path}")
    print(f"Saved {configuration_path}")
    print(f"Saved {protocol_path}")


if __name__ == "__main__":
    main()
