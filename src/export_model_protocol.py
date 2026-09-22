from __future__ import annotations

import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from config import OUTPUTS_TABLES_DIR
    from modeling import (
        _build_gradient_boosting_model,
        _build_logistic_pipeline,
        _build_random_forest_model,
        _build_xgb_model,
        _build_xgb_model_with_balance,
    )
except ImportError:  # pragma: no cover - package import fallback
    from .config import OUTPUTS_TABLES_DIR
    from .modeling import (
        _build_gradient_boosting_model,
        _build_logistic_pipeline,
        _build_random_forest_model,
        _build_xgb_model,
        _build_xgb_model_with_balance,
    )


def package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not_installed"


def normalize_value(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, default=str, sort_keys=True)
    return str(value)


def environment_rows() -> list[dict[str, str]]:
    packages = [
        ("python", sys.version.replace("\n", " ")),
        ("platform", platform.platform()),
        ("scikit-learn", package_version("scikit-learn")),
        ("xgboost", package_version("xgboost")),
        ("numpy", package_version("numpy")),
        ("pandas", package_version("pandas")),
        ("scipy", package_version("scipy")),
        ("joblib", package_version("joblib")),
        ("matplotlib", package_version("matplotlib")),
        ("seaborn", package_version("seaborn")),
        ("wfdb", package_version("wfdb")),
    ]
    return [{"component": name, "version": value} for name, value in packages]


def build_linkability_logistic_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ]
    )


def model_objects() -> list[tuple[str, str, Any, str]]:
    feature_columns = ["feature_001", "feature_002"]
    return [
        (
            "utility_baseline",
            "LogisticRegression",
            _build_logistic_pipeline(feature_columns),
            "src.modeling._build_logistic_pipeline",
        ),
        (
            "utility_baseline",
            "RandomForest",
            _build_random_forest_model(),
            "src.modeling._build_random_forest_model",
        ),
        (
            "utility_baseline",
            "GradientBoosting",
            _build_gradient_boosting_model(),
            "src.modeling._build_gradient_boosting_model",
        ),
        (
            "utility_baseline",
            "XGBoost",
            _build_xgb_model(),
            "src.modeling._build_xgb_model",
        ),
        (
            "utility_tuned_exploratory",
            "XGBoost_tuned_balanced",
            _build_xgb_model_with_balance(scale_pos_weight=1.0),
            "src.modeling._build_xgb_model_with_balance",
        ),
        (
            "linkability_pairwise",
            "LogisticRegression",
            build_linkability_logistic_pipeline(),
            "src.modeling.run_linkability_baselines",
        ),
        (
            "linkability_pairwise",
            "XGBoost",
            _build_xgb_model(),
            "src.modeling._build_xgb_model",
        ),
        (
            "operational_linkage",
            "XGBoost",
            _build_xgb_model(),
            "src.run_operational_linkage_eval.evaluate_seed",
        ),
    ]


def extract_hyperparameter_rows() -> list[dict[str, str]]:
    rows = []
    for scope, model_name, estimator, source in model_objects():
        if isinstance(estimator, Pipeline):
            for step_name, step in estimator.named_steps.items():
                if hasattr(step, "get_params"):
                    params = step.get_params(deep=False)
                else:
                    params = {}
                rows.append(
                    {
                        "protocol_scope": scope,
                        "model_name": model_name,
                        "pipeline_step": step_name,
                        "parameter": "estimator_class",
                        "value": step.__class__.__name__,
                        "source": source,
                    }
                )
                for parameter, value in sorted(params.items()):
                    rows.append(
                        {
                            "protocol_scope": scope,
                            "model_name": model_name,
                            "pipeline_step": step_name,
                            "parameter": parameter,
                            "value": normalize_value(value),
                            "source": source,
                        }
                    )
        else:
            rows.append(
                {
                    "protocol_scope": scope,
                    "model_name": model_name,
                    "pipeline_step": "model",
                    "parameter": "estimator_class",
                    "value": estimator.__class__.__name__,
                    "source": source,
                }
            )
            for parameter, value in sorted(estimator.get_params().items()):
                rows.append(
                    {
                        "protocol_scope": scope,
                        "model_name": model_name,
                        "pipeline_step": "model",
                        "parameter": parameter,
                        "value": normalize_value(value),
                        "source": source,
                    }
                )
    return rows


def protocol_summary_rows() -> list[dict[str, str]]:
    return [
        {
            "protocol_scope": "utility_baseline",
            "task": "utility",
            "models": "LogisticRegression, RandomForest, GradientBoosting, XGBoost",
            "split_protocol": "patient-level stratified holdout over unique patient_id/utility_label; test_size=0.2; default random_state=42 unless a repeated-seed experiment overrides it",
            "folds": "none in current src implementation; no StratifiedGroupKFold is used",
            "feature_preprocessing": "LogisticRegression uses train-fitted SimpleImputer(strategy='median') + StandardScaler in Pipeline; tree/XGBoost models use train-fitted SimpleImputer(strategy='median') and no scaling",
            "imbalance_handling": "LogisticRegression and RandomForest use class_weight='balanced'; baseline XGBoost/GradientBoosting use no explicit class weighting",
            "threshold": "0.5 for baseline predictions",
            "seeds": "split seed defaults to 42; estimator random_state=42 where supported",
            "source": "src/modeling.py::run_utility_baselines",
        },
        {
            "protocol_scope": "utility_tuned_exploratory",
            "task": "utility",
            "models": "LogisticRegression_tuned, RandomForest_tuned, GradientBoosting_tuned, XGBoost_tuned",
            "split_protocol": "patient-level train/validation/test split; test_size=0.2; validation occupies 0.2 of full data via second split with random_state+1",
            "folds": "none; validation holdout only",
            "feature_preprocessing": "same preprocessing as utility_baseline",
            "imbalance_handling": "LogisticRegression/RandomForest use class_weight='balanced'; XGBoost_tuned uses scale_pos_weight=negatives/positives computed on the training split and recomputed on train+validation",
            "threshold": "threshold selected on validation F1 over np.linspace(0.05, 0.95, 19)",
            "seeds": "default split random_state=42; estimator random_state=42",
            "source": "src/modeling.py::run_utility_baselines_tuned",
        },
        {
            "protocol_scope": "linkability_pairwise",
            "task": "linkability",
            "models": "LogisticRegression, XGBoost; optional EuclideanDistance/CosineSimilarity baselines",
            "split_protocol": "patient-level holdout; run_linkability_baselines uses GroupShuffleSplit with test_size=0.2; run_linkability_baselines_on_split reuses an externally supplied patient split",
            "folds": "none; repeated robustness experiments use seeds [42, 123, 456]",
            "feature_preprocessing": "pair representation is absdiff by default; LogisticRegression applies StandardScaler to pair features; XGBoost uses raw pair features; distance baselines impute and standardize segment features before computing similarity",
            "imbalance_handling": "pair labels are balanced by construction; LogisticRegression also uses class_weight='balanced'; XGBoost has no scale_pos_weight in pairwise baseline",
            "threshold": "0.5 for model baselines; distance baselines choose threshold on training pairs",
            "seeds": "split/pair seed defaults to 42; test pairs use random_state+1; estimator random_state=42",
            "source": "src/modeling.py::run_linkability_baselines and run_linkability_baselines_on_split",
        },
        {
            "protocol_scope": "privacy_transform_final",
            "task": "utility_and_linkability",
            "models": "Utility LogisticRegression; Linkability XGBoost",
            "split_protocol": "patient-level stratified holdout per seed; test_size=0.2; seeds [42, 123, 456]",
            "folds": "none; repeated holdout over three seeds",
            "feature_preprocessing": "privacy transform fitted on train feature matrix after median imputation and applied to train/test; linkability pairs are built after transformation",
            "imbalance_handling": "utility LogisticRegression class_weight='balanced'; linkability pairs balanced by construction",
            "threshold": "0.5 for both models",
            "seeds": "transforms and splits use seeds [42, 123, 456]; noise test split uses seed+10000 offset",
            "source": "src/run_privacy_transform_study.py",
        },
        {
            "protocol_scope": "operational_linkage",
            "task": "linkability",
            "models": "XGBoost",
            "split_protocol": "patient-level stratified holdout per seed; train pair data sampled from train split; synthetic gallery sampled from test split",
            "folds": "none; repeated holdout over seeds [42, 123, 456]",
            "feature_preprocessing": "absdiff pair features; StandardScaler fitted on train pairs and applied to gallery pairs before XGBoost",
            "imbalance_handling": "train pairs balanced by construction; gallery prevalence is synthetic and low",
            "threshold": "ranking/score metrics; no fixed classification threshold for Recall@k/MRR",
            "seeds": "seeds [42, 123, 456]; gallery sampling uses seed+gallery_negatives",
            "source": "src/run_operational_linkage_eval.py",
        },
    ]


def main() -> None:
    OUTPUTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    environment_path = OUTPUTS_TABLES_DIR / "computational_environment_versions.csv"
    summary_path = OUTPUTS_TABLES_DIR / "model_computational_protocol_summary.csv"
    hyperparams_path = OUTPUTS_TABLES_DIR / "model_hyperparameters.csv"

    pd.DataFrame(environment_rows()).to_csv(environment_path, index=False)
    pd.DataFrame(protocol_summary_rows()).to_csv(summary_path, index=False)
    pd.DataFrame(extract_hyperparameter_rows()).to_csv(hyperparams_path, index=False)

    print("Saved environment:", environment_path)
    print("Saved protocol summary:", summary_path)
    print("Saved hyperparameters:", hyperparams_path)


if __name__ == "__main__":
    main()
