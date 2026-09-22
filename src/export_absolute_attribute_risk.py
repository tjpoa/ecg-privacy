"""Export absolute attribute-inference performance and no-skill baselines."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
SOURCE = TABLES / "attribute_inference_final_protocol_multiseed_metrics.csv"
SEEDS = [42, 123, 456]


VIEWS = [
    {
        "view_order": 1,
        "view": "Constant / training-median baseline",
        "short_label": "No-information baseline",
        "representation": "constant_baseline",
        "train_condition": "single_segment",
        "test_condition": "single_segment",
    },
    {
        "view_order": 2,
        "view": "Utility-label-only baseline",
        "short_label": "Utility label only",
        "representation": "utility_only",
        "train_condition": "single_segment",
        "test_condition": "single_segment",
    },
    {
        "view_order": 3,
        "view": "Raw handcrafted, single segment",
        "short_label": "Raw, single",
        "representation": "raw_features",
        "train_condition": "single_segment",
        "test_condition": "single_segment",
    },
    {
        "view_order": 4,
        "view": "Raw handcrafted, attacker-linked aggregate",
        "short_label": "Raw, linked",
        "representation": "raw_features",
        "train_condition": "oracle_group",
        "test_condition": "link_selected_group",
    },
    {
        "view_order": 5,
        "view": "Supervised encoder, single segment",
        "short_label": "Encoder, single",
        "representation": "encoder",
        "train_condition": "single_segment",
        "test_condition": "single_segment",
    },
    {
        "view_order": 6,
        "view": "Supervised encoder, attacker-linked aggregate",
        "short_label": "Encoder, linked",
        "representation": "encoder",
        "train_condition": "oracle_group",
        "test_condition": "link_selected_group",
    },
    {
        "view_order": 7,
        "view": "Gaussian C=2, single segment",
        "short_label": "Gaussian C=2, single",
        "representation": "encoder_gaussian_noise",
        "train_condition": "single_segment",
        "test_condition": "single_segment",
    },
    {
        "view_order": 8,
        "view": "Gaussian C=2, attacker-linked aggregate",
        "short_label": "Gaussian C=2, linked",
        "representation": "encoder_gaussian_noise",
        "train_condition": "oracle_group",
        "test_condition": "link_selected_group",
    },
]


def select_task_rows(metrics: pd.DataFrame, view: dict[str, object], task: str) -> pd.DataFrame:
    selected = metrics[
        metrics["task"].eq(task)
        & metrics["representation"].eq(view["representation"])
        & metrics["train_condition"].eq(view["train_condition"])
        & metrics["test_condition"].eq(view["test_condition"])
    ].copy()
    if sorted(selected["seed"].astype(int).tolist()) != SEEDS:
        raise ValueError(f"Expected one row per seed for {view['view']} / {task}")
    return selected.set_index("seed").sort_index()


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    metrics = pd.read_csv(SOURCE)

    seed_rows: list[dict[str, object]] = []
    for view in VIEWS:
        sex = select_task_rows(metrics, view, "sex")
        age_group = select_task_rows(metrics, view, "age_group")
        age = select_task_rows(metrics, view, "age")
        for seed in SEEDS:
            seed_rows.append(
                {
                    **view,
                    "seed": seed,
                    "sex_balanced_accuracy": float(sex.loc[seed, "balanced_accuracy"]),
                    "age_group_balanced_accuracy": float(age_group.loc[seed, "balanced_accuracy"]),
                    "age_mae_years": float(age.loc[seed, "mae_years"]),
                    "sex_n_test": int(sex.loc[seed, "n_test"]),
                    "age_n_test": int(age.loc[seed, "n_test"]),
                }
            )

    seed_frame = pd.DataFrame(seed_rows).sort_values(["view_order", "seed"])
    summary = (
        seed_frame.groupby(
            [
                "view_order",
                "view",
                "short_label",
                "representation",
                "train_condition",
                "test_condition",
            ],
            as_index=False,
        )
        .agg(
            n_seeds=("seed", "nunique"),
            sex_balanced_accuracy_mean=("sex_balanced_accuracy", "mean"),
            sex_balanced_accuracy_std=("sex_balanced_accuracy", "std"),
            age_group_balanced_accuracy_mean=("age_group_balanced_accuracy", "mean"),
            age_group_balanced_accuracy_std=("age_group_balanced_accuracy", "std"),
            age_mae_years_mean=("age_mae_years", "mean"),
            age_mae_years_std=("age_mae_years", "std"),
            sex_n_test_mean=("sex_n_test", "mean"),
            age_n_test_mean=("age_n_test", "mean"),
        )
        .sort_values("view_order")
    )

    constant = summary.loc[summary["view_order"].eq(1)].iloc[0]
    utility = summary.loc[summary["view_order"].eq(2)].iloc[0]
    summary["sex_ba_excess_over_constant"] = (
        summary["sex_balanced_accuracy_mean"] - constant["sex_balanced_accuracy_mean"]
    )
    summary["age_group_ba_excess_over_constant"] = (
        summary["age_group_balanced_accuracy_mean"]
        - constant["age_group_balanced_accuracy_mean"]
    )
    summary["age_mae_improvement_over_constant_years"] = (
        constant["age_mae_years_mean"] - summary["age_mae_years_mean"]
    )
    summary["sex_ba_excess_over_utility_only"] = (
        summary["sex_balanced_accuracy_mean"] - utility["sex_balanced_accuracy_mean"]
    )
    summary["age_group_ba_excess_over_utility_only"] = (
        summary["age_group_balanced_accuracy_mean"]
        - utility["age_group_balanced_accuracy_mean"]
    )
    summary["age_mae_improvement_over_utility_only_years"] = (
        utility["age_mae_years_mean"] - summary["age_mae_years_mean"]
    )

    seed_path = TABLES / "absolute_attribute_risk_seed_metrics.csv"
    summary_path = TABLES / "absolute_attribute_risk_summary.csv"
    protocol_path = TABLES / "absolute_attribute_risk_protocol.json"
    seed_frame.to_csv(seed_path, index=False)
    summary.to_csv(summary_path, index=False)

    protocol = {
        "purpose": (
            "Report absolute attribute-inference performance, rather than only the change "
            "caused by attacker-selected aggregation."
        ),
        "classification_baseline": (
            "Predict the majority class estimated on training records. Balanced accuracy is "
            "0.5 for this constant predictor."
        ),
        "continuous_age_baseline": (
            "Predict the median age estimated on training records and report test MAE in years."
        ),
        "utility_only_baseline": (
            "Train the same fixed-capacity attribute attacker using only the binary, "
            "record-derived clinical utility label as its input."
        ),
        "single_view": "attacker trained and tested on one segment per retained record",
        "linked_view": (
            "attacker trained on oracle same-record aggregates and tested on aggregates "
            "selected by the linkability attacker"
        ),
        "metrics": {
            "sex": "balanced accuracy; Female is positive",
            "age_group": "balanced accuracy; age >= 65 years is positive",
            "continuous_age": "mean absolute error in years; lower is stronger inference",
        },
        "seeds": SEEDS,
        "source": SOURCE.name,
        "outputs": [seed_path.name, summary_path.name],
    }
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"Saved {seed_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {protocol_path}")


if __name__ == "__main__":
    main()
