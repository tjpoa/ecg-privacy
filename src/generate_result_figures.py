from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

try:
    from config import FEATURE_SETS_DIR, OUTPUTS_FIGURES_DIR, OUTPUTS_TABLES_DIR
except ImportError:  # pragma: no cover - package import fallback
    from .config import FEATURE_SETS_DIR, OUTPUTS_FIGURES_DIR, OUTPUTS_TABLES_DIR


def _save_current_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def build_segmentation_selection_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "config": "w2_o0p5",
                "window_sec": 2.0,
                "step_sec": 1.0,
                "utility_logreg_f1": 0.723567,
                "linkability_xgb_roc_auc": 0.998898,
                "decision": "selected",
            },
            {
                "config": "w3_o0p5",
                "window_sec": 3.0,
                "step_sec": 1.5,
                "utility_logreg_f1": 0.737304,
                "linkability_xgb_roc_auc": 0.999357,
                "decision": "comparison",
            },
        ]
    )


def build_baseline_results_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"task": "Utility", "model": "LogisticRegression", "f1_score": 0.678900, "roc_auc": 0.907671},
            {"task": "Utility", "model": "XGBoost", "f1_score": 0.675260, "roc_auc": 0.925902},
            {"task": "Linkability", "model": "LogisticRegression", "f1_score": 0.975219, "roc_auc": 0.996190},
            {"task": "Linkability", "model": "XGBoost", "f1_score": 0.978809, "roc_auc": 0.998343},
        ]
    )


def build_feature_removal_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"n_removed": 0, "utility_f1": 0.678106, "linkability_roc_auc": 0.997051},
            {"n_removed": 10, "utility_f1": 0.676987, "linkability_roc_auc": 0.996015},
            {"n_removed": 20, "utility_f1": 0.665033, "linkability_roc_auc": 0.995232},
            {"n_removed": 30, "utility_f1": 0.653883, "linkability_roc_auc": 0.994811},
            {"n_removed": 40, "utility_f1": 0.646042, "linkability_roc_auc": 0.993679},
            {"n_removed": 50, "utility_f1": 0.635907, "linkability_roc_auc": 0.993234},
        ]
    )


def plot_segmentation_selection(out_dir: Path) -> None:
    df = build_segmentation_selection_df()
    plt.figure(figsize=(6, 4))
    colors = {"selected": "#1b9e77", "comparison": "#d95f02"}

    for _, row in df.iterrows():
        plt.scatter(
            row["linkability_xgb_roc_auc"],
            row["utility_logreg_f1"],
            s=90,
            color=colors[row["decision"]],
        )
        plt.annotate(row["config"], (row["linkability_xgb_roc_auc"], row["utility_logreg_f1"]), xytext=(6, 6), textcoords="offset points")

    plt.xlabel("Linkability ROC-AUC (XGBoost)")
    plt.ylabel("Utility F1 (LogisticRegression)")
    plt.title("Segmentation Selection")
    _save_current_figure(out_dir / "segmentation_selection_tradeoff.png")


def plot_baseline_comparison(out_dir: Path) -> None:
    df = build_baseline_results_df()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    for ax, metric, title in zip(
        axes,
        ["f1_score", "roc_auc"],
        ["Baseline F1", "Baseline ROC-AUC"],
    ):
        pivot = df.pivot(index="task", columns="model", values=metric)
        pivot.plot(kind="bar", ax=ax, rot=0)
        ax.set_title(title)
        ax.set_ylabel(metric.replace("_", " ").upper())
        ax.legend(title="Model")

    _save_current_figure(out_dir / "baseline_model_comparison.png")


def plot_utility_feature_selection(out_dir: Path) -> None:
    summary_path = FEATURE_SETS_DIR / "utility_feature_selection_summary.csv"
    df = pd.read_csv(summary_path)
    df = df[df["model"] == "LogisticRegression"].sort_values("subset_size")

    plt.figure(figsize=(7, 4))
    plt.plot(df["subset_size"], df["f1_score"], marker="o", label="F1")
    plt.plot(df["subset_size"], df["balanced_accuracy"], marker="s", label="Balanced Accuracy")
    plt.axhline(0.80, color="gray", linestyle="--", linewidth=1, label="BA target = 0.80")
    plt.xlabel("Number of Features")
    plt.ylabel("Score")
    plt.title("Utility vs Number of Features")
    plt.legend()
    _save_current_figure(out_dir / "utility_feature_selection_curve.png")


def plot_feature_removal_tradeoff(out_dir: Path) -> None:
    df = build_feature_removal_df()
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()

    ax1.plot(df["n_removed"], df["utility_f1"], color="#1f77b4", marker="o", label="Utility F1")
    ax2.plot(df["n_removed"], df["linkability_roc_auc"], color="#d62728", marker="s", label="Linkability ROC-AUC")

    ax1.set_xlabel("Removed Features")
    ax1.set_ylabel("Utility F1", color="#1f77b4")
    ax2.set_ylabel("Linkability ROC-AUC", color="#d62728")
    ax1.set_title("Progressive Feature Removal Trade-off")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")
    _save_current_figure(out_dir / "feature_removal_tradeoff.png")


def plot_privacy_transformations(out_dir: Path) -> None:
    df = pd.read_csv(OUTPUTS_TABLES_DIR / "privacy_transformations_summary.csv")
    plt.figure(figsize=(7, 4))
    plt.scatter(df["linkability_roc_auc"], df["utility_f1"], s=80, color="#4c78a8")
    for _, row in df.iterrows():
        plt.annotate(row["transform_name"], (row["linkability_roc_auc"], row["utility_f1"]), xytext=(5, 4), textcoords="offset points")

    plt.xlabel("Linkability ROC-AUC")
    plt.ylabel("Utility F1")
    plt.title("Privacy Transformations Trade-off")
    _save_current_figure(out_dir / "privacy_transformations_tradeoff.png")


def main() -> None:
    OUTPUTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_segmentation_selection(OUTPUTS_FIGURES_DIR)
    plot_baseline_comparison(OUTPUTS_FIGURES_DIR)
    plot_utility_feature_selection(OUTPUTS_FIGURES_DIR)
    plot_feature_removal_tradeoff(OUTPUTS_FIGURES_DIR)
    plot_privacy_transformations(OUTPUTS_FIGURES_DIR)
    print(f"Saved figures to: {OUTPUTS_FIGURES_DIR}")


if __name__ == "__main__":
    main()
