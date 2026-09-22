from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from config import OUTPUTS_FIGURES_DIR, OUTPUTS_TABLES_DIR
except ImportError:  # pragma: no cover - package import fallback
    from .config import OUTPUTS_FIGURES_DIR, OUTPUTS_TABLES_DIR


def _label_representation(method: str, dimension: int | float | None) -> str:
    if method == "identity":
        return "raw 208 features"
    if method == "pca":
        return f"PCA {int(dimension)}"
    if method == "random_projection":
        return f"Random projection {int(dimension)}"
    if method == "topk_utility":
        return f"Top-k utility {int(dimension)}"
    if method == "supervised_encoder":
        return f"Supervised encoder {int(dimension)}"
    return str(method)


def _gaussian_label(noise_sigma: float, clip_norm: float) -> str:
    noise_multiplier = float(noise_sigma) / float(clip_norm)
    return f"Encoder dim8 + Gaussian m={noise_multiplier:.3f}, C={float(clip_norm):g}"


def build_broad_representation_table() -> pd.DataFrame:
    multiseed_path = OUTPUTS_TABLES_DIR / "representation_dimension_final_multiseed_summary.csv"
    legacy_path = OUTPUTS_TABLES_DIR / "representation_dimension_comparison_seed42_full_summary.csv"
    representation_path = multiseed_path if multiseed_path.exists() else legacy_path
    evidence_scope = "three_seed_broad_comparison" if multiseed_path.exists() else "seed42_broad_comparison"
    encoder_dp_path = OUTPUTS_TABLES_DIR / "encoder_dp_dim8_seed42_summary.csv"
    operational_path = OUTPUTS_TABLES_DIR / "operational_linkage_encoder_dp_comparison.csv"

    representation_df = pd.read_csv(representation_path)
    rows = []
    for _, row in representation_df.iterrows():
        method = row["method"]
        dimension = int(row["dimension"])
        rows.append(
            {
                "evidence_scope": evidence_scope,
                "representation_family": method,
                "representation": _label_representation(method, dimension),
                "dimension": dimension,
                "dp_epsilon": np.nan,
                "dp_clip_norm": np.nan,
                "noise_multiplier": np.nan,
                "utility_f1": row["utility_f1_mean"],
                "utility_f1_std": row.get("utility_f1_std", np.nan),
                "utility_balanced_accuracy": row["utility_balanced_accuracy_mean"],
                "utility_balanced_accuracy_std": row.get("utility_balanced_accuracy_std", np.nan),
                "utility_roc_auc": row["utility_roc_auc_mean"],
                "utility_pr_auc": row["utility_pr_auc_mean"],
                "pairwise_linkability_roc_auc": row["linkability_roc_auc_mean"],
                "pairwise_linkability_roc_auc_std": row.get("linkability_roc_auc_std", np.nan),
                "pairwise_linkability_pr_auc": row["linkability_pr_auc_mean"],
                "pairwise_linkability_pr_auc_std": row.get("linkability_pr_auc_std", np.nan),
                "operational_gallery_negatives": np.nan,
                "operational_positive_prevalence": np.nan,
                "operational_pr_auc": np.nan,
                "operational_recall_at_1": np.nan,
                "operational_mrr": np.nan,
            }
        )

    encoder_dp_df = pd.read_csv(encoder_dp_path)
    for _, row in encoder_dp_df.iterrows():
        epsilon = row["dp_epsilon"]
        clip_norm = row["dp_clip_norm"]
        rows.append(
            {
                "evidence_scope": "seed42_encoder_dp",
                "representation_family": "encoder_dp",
                "representation": _gaussian_label(row["dp_noise_sigma"], clip_norm),
                "dimension": int(row["n_output_features"]),
                "dp_epsilon": epsilon,
                "dp_clip_norm": clip_norm,
                "noise_multiplier": row["dp_noise_sigma"] / clip_norm,
                "utility_f1": row["utility_f1_mean"],
                "utility_balanced_accuracy": row["utility_balanced_accuracy_mean"],
                "utility_roc_auc": row["utility_roc_auc_mean"],
                "utility_pr_auc": row["utility_pr_auc_mean"],
                "pairwise_linkability_roc_auc": row["linkability_roc_auc_mean"],
                "pairwise_linkability_pr_auc": row["linkability_pr_auc_mean"],
                "operational_gallery_negatives": np.nan,
                "operational_positive_prevalence": np.nan,
                "operational_pr_auc": np.nan,
                "operational_recall_at_1": np.nan,
                "operational_mrr": np.nan,
            }
        )

    out = pd.DataFrame(rows)

    if operational_path.exists():
        operational_df = pd.read_csv(operational_path)
        operational_999 = operational_df[operational_df["gallery_negatives"].eq(999)].copy()
        operational_map = {
            "raw 208 features": "raw 208 features",
            "encoder dim8": "Supervised encoder 8",
            "encoder+DP eps100 c2": "Encoder dim8 + Gaussian m=0.097, C=2",
            "encoder+DP eps50 c2": "Encoder dim8 + Gaussian m=0.194, C=2",
            "encoder+DP eps50 c4": "Encoder dim8 + Gaussian m=0.194, C=4",
            "encoder+DP eps20 c4": "Encoder dim8 + Gaussian m=0.484, C=4",
        }
        operational_999["representation"] = operational_999["transform_label"].map(operational_map)
        metric_cols = [
            "operational_positive_prevalence",
            "operational_pr_auc",
            "operational_recall_at_1",
            "operational_mrr",
        ]
        for _, op_row in operational_999.dropna(subset=["representation"]).iterrows():
            mask = out["representation"].eq(op_row["representation"])
            out.loc[mask, "operational_gallery_negatives"] = int(op_row["gallery_negatives"])
            out.loc[mask, "operational_positive_prevalence"] = op_row["positive_prevalence"]
            out.loc[mask, "operational_pr_auc"] = op_row["link_operational_pr_auc"]
            out.loc[mask, "operational_recall_at_1"] = op_row["recall_at_1"]
            out.loc[mask, "operational_mrr"] = op_row["mrr"]
        out[metric_cols] = out[metric_cols].apply(pd.to_numeric, errors="coerce")

    return out


def build_encoder_dp_multiseed_table() -> pd.DataFrame:
    path = OUTPUTS_TABLES_DIR / "encoder_dp_operational_dim8_multiseed_summary.csv"
    df = pd.read_csv(path)
    df = df[df["gallery_negatives"].eq(999)].copy()
    df["noise_multiplier"] = df["dp_noise_sigma"] / df["dp_clip_norm"]
    df["representation"] = [
        "Encoder dim8"
        if transform_name == "encoder_identity"
        else _gaussian_label(noise_sigma, clip_norm)
        for transform_name, noise_sigma, clip_norm in zip(
            df["transform_name"], df["dp_noise_sigma"], df["dp_clip_norm"]
        )
    ]
    selected_cols = [
        "representation",
        "transform_name",
        "bottleneck_dim",
        "n_output_features",
        "dp_epsilon",
        "dp_delta",
        "dp_clip_norm",
        "dp_noise_sigma",
        "noise_multiplier",
        "utility_f1_mean",
        "utility_f1_std",
        "utility_balanced_accuracy_mean",
        "utility_balanced_accuracy_std",
        "utility_roc_auc_mean",
        "utility_roc_auc_std",
        "pairwise_linkability_roc_auc_mean",
        "pairwise_linkability_roc_auc_std",
        "pairwise_linkability_pr_auc_mean",
        "pairwise_linkability_pr_auc_std",
        "pairwise_linkability_eer_mean",
        "pairwise_linkability_eer_std",
        "gallery_negatives",
        "positive_prevalence_mean",
        "link_operational_roc_auc_mean",
        "link_operational_roc_auc_std",
        "link_operational_pr_auc_mean",
        "link_operational_pr_auc_std",
        "link_operational_eer_mean",
        "link_operational_eer_std",
        "recall_at_1_mean",
        "recall_at_1_std",
        "recall_at_5_mean",
        "recall_at_5_std",
        "mrr_mean",
        "mrr_std",
        "tpr_at_fpr_0.001_mean",
        "tpr_at_fpr_0.001_std",
    ]
    return df[selected_cols].sort_values(
        ["dp_clip_norm", "noise_multiplier"],
        na_position="first",
    )


def build_thesis_ready_table(broad_df: pd.DataFrame, multiseed_df: pd.DataFrame) -> pd.DataFrame:
    seed42_labels = {
        "raw 208 features": "Raw handcrafted features",
        "PCA 64": "PCA 64 components",
        "Random projection 64": "Random projection 64 components",
    }
    rows = []
    for representation, label in seed42_labels.items():
        match = broad_df[broad_df["representation"].eq(representation)]
        if match.empty:
            continue
        row = match.iloc[0]
        evidence = "three-seed retained-record-level representation comparison"
        if pd.notna(row.get("operational_pr_auc", np.nan)):
            evidence += "; seed-42 operational gallery"
        rows.append(
            {
                "representation": label,
                "evidence": evidence,
                "dimension": row["dimension"],
                "dp_epsilon": np.nan,
                "dp_clip_norm": np.nan,
                "noise_multiplier": np.nan,
                "utility_balanced_accuracy_mean": row["utility_balanced_accuracy"],
                "utility_balanced_accuracy_std": row.get("utility_balanced_accuracy_std", np.nan),
                "utility_f1_mean": row["utility_f1"],
                "utility_f1_std": row.get("utility_f1_std", np.nan),
                "pairwise_linkability_roc_auc_mean": row["pairwise_linkability_roc_auc"],
                "pairwise_linkability_roc_auc_std": row.get("pairwise_linkability_roc_auc_std", np.nan),
                "pairwise_linkability_eer_mean": np.nan,
                "pairwise_linkability_eer_std": np.nan,
                "operational_pr_auc_mean": row["operational_pr_auc"],
                "operational_pr_auc_std": np.nan,
                "operational_recall_at_1_mean": row["operational_recall_at_1"],
                "operational_recall_at_1_std": np.nan,
                "operational_eer_mean": np.nan,
                "operational_eer_std": np.nan,
                "operational_mrr_mean": row["operational_mrr"],
                "operational_mrr_std": np.nan,
            }
        )

    final_labels = {
        "Encoder dim8": "Supervised encoder dim8",
        "Encoder dim8 + Gaussian m=0.194, C=2": "Encoder dim8 + Gaussian m=0.194, C=2",
        "Encoder dim8 + Gaussian m=0.194, C=4": "Encoder dim8 + Gaussian m=0.194, C=4",
    }
    for representation, label in final_labels.items():
        match = multiseed_df[multiseed_df["representation"].eq(representation)]
        if match.empty:
            continue
        row = match.iloc[0]
        rows.append(
            {
                "representation": label,
                "evidence": "three-seed retained-record-level split",
                "dimension": row["n_output_features"],
                "dp_epsilon": row["dp_epsilon"],
                "dp_clip_norm": row["dp_clip_norm"],
                "noise_multiplier": row["noise_multiplier"],
                "utility_balanced_accuracy_mean": row["utility_balanced_accuracy_mean"],
                "utility_balanced_accuracy_std": row["utility_balanced_accuracy_std"],
                "utility_f1_mean": row["utility_f1_mean"],
                "utility_f1_std": row["utility_f1_std"],
                "pairwise_linkability_roc_auc_mean": row["pairwise_linkability_roc_auc_mean"],
                "pairwise_linkability_roc_auc_std": row["pairwise_linkability_roc_auc_std"],
                "pairwise_linkability_eer_mean": row["pairwise_linkability_eer_mean"],
                "pairwise_linkability_eer_std": row["pairwise_linkability_eer_std"],
                "operational_pr_auc_mean": row["link_operational_pr_auc_mean"],
                "operational_pr_auc_std": row["link_operational_pr_auc_std"],
                "operational_recall_at_1_mean": row["recall_at_1_mean"],
                "operational_recall_at_1_std": row["recall_at_1_std"],
                "operational_eer_mean": row["link_operational_eer_mean"],
                "operational_eer_std": row["link_operational_eer_std"],
                "operational_mrr_mean": row["mrr_mean"],
                "operational_mrr_std": row["mrr_std"],
            }
        )

    return pd.DataFrame(rows)


def save_broad_pairwise_figure(broad_df: pd.DataFrame) -> Path:
    OUTPUTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 5.5))
    families = [
        "identity",
        "pca",
        "random_projection",
        "topk_utility",
        "supervised_encoder",
        "encoder_dp",
    ]
    colors = {
        "identity": "#222222",
        "pca": "#457b9d",
        "random_projection": "#2a9d8f",
        "topk_utility": "#e9c46a",
        "supervised_encoder": "#e76f51",
        "encoder_dp": "#8d5a97",
    }
    family_labels = {
        "identity": "Raw features",
        "pca": "PCA",
        "random_projection": "Random projection",
        "topk_utility": "Top-k utility",
        "supervised_encoder": "Supervised encoder",
        "encoder_dp": "Gaussian perturbation",
    }
    for family in families:
        group = broad_df[broad_df["representation_family"].eq(family)]
        if group.empty:
            continue
        ax.scatter(
            group["pairwise_linkability_roc_auc"],
            group["utility_balanced_accuracy"],
            s=70,
            label=family_labels.get(family, family),
            color=colors.get(family),
            alpha=0.85,
        )
    highlight = broad_df[
        broad_df["representation"].isin(
            [
                "raw 208 features",
                "Supervised encoder 8",
                "Encoder dim8 + Gaussian m=0.194, C=2",
                "Encoder dim8 + Gaussian m=0.194, C=4",
            ]
        )
    ]
    annotation_labels = {
        "raw 208 features": "Raw 208 features",
        "Supervised encoder 8": "Encoder 8",
        "Encoder dim8 + Gaussian m=0.194, C=2": "Gaussian $m=0.194,C=2$",
        "Encoder dim8 + Gaussian m=0.194, C=4": "Gaussian $m=0.194,C=4$",
    }
    for _, row in highlight.iterrows():
        ax.annotate(
            annotation_labels.get(row["representation"], row["representation"]),
            (row["pairwise_linkability_roc_auc"], row["utility_balanced_accuracy"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Pairwise linkability ROC-AUC")
    ax.set_ylabel("Utility balanced accuracy")
    ax.set_title("Representation utility-linkability trade-off")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = OUTPUTS_FIGURES_DIR / "final_representation_pairwise_tradeoff_multiseed.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    return out


def save_multiseed_operational_figure(multiseed_df: pd.DataFrame) -> Path:
    OUTPUTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_df = multiseed_df.copy()
    order = [
        "Encoder dim8",
        "Encoder dim8 + Gaussian m=0.097, C=2",
        "Encoder dim8 + Gaussian m=0.194, C=2",
        "Encoder dim8 + Gaussian m=0.194, C=4",
        "Encoder dim8 + Gaussian m=0.484, C=4",
    ]
    plot_df["representation"] = pd.Categorical(plot_df["representation"], categories=order, ordered=True)
    plot_df = plot_df.sort_values("representation")
    plot_labels = {
        "Encoder dim8": "Encoder 8",
        "Encoder dim8 + Gaussian m=0.097, C=2": "$m=0.097,C=2$",
        "Encoder dim8 + Gaussian m=0.194, C=2": "$m=0.194,C=2$",
        "Encoder dim8 + Gaussian m=0.194, C=4": "$m=0.194,C=4$",
        "Encoder dim8 + Gaussian m=0.484, C=4": "$m=0.484,C=4$",
    }
    plot_df["plot_label"] = plot_df["representation"].astype(str).map(plot_labels)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].bar(
        plot_df["plot_label"],
        plot_df["link_operational_pr_auc_mean"],
        yerr=plot_df["link_operational_pr_auc_std"].fillna(0.0),
        color="#457b9d",
        capsize=3,
    )
    axes[0].set_title("Operational PR-AUC, 0.1% prevalence")
    axes[0].set_ylabel("PR-AUC")
    axes[0].set_ylim(0, max(0.05, plot_df["link_operational_pr_auc_mean"].max() * 1.35))
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(
        plot_df["plot_label"],
        plot_df["recall_at_1_mean"],
        yerr=plot_df["recall_at_1_std"].fillna(0.0),
        color="#e76f51",
        capsize=3,
    )
    axes[1].set_title("Recall@1, 0.1% prevalence")
    axes[1].set_ylabel("Recall@1")
    axes[1].set_ylim(0, max(0.1, plot_df["recall_at_1_mean"].max() * 1.35))
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = OUTPUTS_FIGURES_DIR / "final_encoder_dp_operational_multiseed.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    return out


def main() -> None:
    OUTPUTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    broad_df = build_broad_representation_table()
    multiseed_df = build_encoder_dp_multiseed_table()
    thesis_ready_df = build_thesis_ready_table(broad_df, multiseed_df)

    broad_path = OUTPUTS_TABLES_DIR / "final_representation_tradeoff_multiseed.csv"
    multiseed_path = OUTPUTS_TABLES_DIR / "final_encoder_dp_multiseed_summary.csv"
    thesis_ready_path = OUTPUTS_TABLES_DIR / "final_thesis_ready_representation_table.csv"
    broad_df.to_csv(broad_path, index=False)
    multiseed_df.to_csv(multiseed_path, index=False)
    thesis_ready_df.to_csv(thesis_ready_path, index=False)

    pairwise_fig = save_broad_pairwise_figure(broad_df)
    operational_fig = save_multiseed_operational_figure(multiseed_df)

    print("Saved:", broad_path)
    print("Saved:", multiseed_path)
    print("Saved:", thesis_ready_path)
    print("Saved:", pairwise_fig)
    print("Saved:", operational_fig)


if __name__ == "__main__":
    main()
