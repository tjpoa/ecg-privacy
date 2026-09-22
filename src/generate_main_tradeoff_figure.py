from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from config import OUTPUTS_FIGURES_DIR, OUTPUTS_TABLES_DIR, PROJECT_ROOT
except ImportError:  # pragma: no cover - package import fallback
    from .config import OUTPUTS_FIGURES_DIR, OUTPUTS_TABLES_DIR, PROJECT_ROOT


PUBLIC_FIGURES_DIR = PROJECT_ROOT / "figures"


def _select_row(
    frame: pd.DataFrame,
    method: str,
    dimension: int,
    evaluation_level: str | None = None,
) -> pd.Series:
    mask = frame["method"].eq(method) & frame["dimension"].eq(dimension)
    if evaluation_level is not None:
        mask &= frame["evaluation_level"].eq(evaluation_level)
    selected = frame.loc[mask]
    if len(selected) != 1:
        raise ValueError(
            "Expected exactly one row for "
            f"method={method!r}, dimension={dimension}, "
            f"evaluation_level={evaluation_level!r}; found {len(selected)}."
        )
    return selected.iloc[0]


def build_figure_data() -> pd.DataFrame:
    """Assemble exactly the six sample-matched rows reported in Table I."""
    matched = pd.read_csv(
        OUTPUTS_TABLES_DIR / "representation_dimension_submission_matched_summary.csv"
    )
    matched_record = pd.read_csv(
        OUTPUTS_TABLES_DIR
        / "representation_dimension_submission_matched_record_level_summary.csv"
    )
    controlled = pd.read_csv(
        OUTPUTS_TABLES_DIR / "article_controlled_representation_multiseed_summary.csv"
    )
    controlled_record = pd.read_csv(
        OUTPUTS_TABLES_DIR / "article_controlled_record_level_utility_summary.csv"
    )

    specifications = [
        ("Raw handcrafted", "identity", 208, matched, matched_record),
        ("PCA, 8-D", "pca", 8, matched, matched_record),
        (
            "Random projection, 8-D",
            "random_projection",
            8,
            matched,
            matched_record,
        ),
        (
            "Supervised encoder, 8-D",
            "supervised_encoder",
            8,
            matched,
            matched_record,
        ),
        (
            "Encoder + Gaussian, C=2",
            "encoder_gaussian_c2",
            8,
            controlled,
            controlled_record,
        ),
        (
            "Encoder + Gaussian, C=4",
            "encoder_gaussian_c4",
            8,
            controlled,
            controlled_record,
        ),
    ]

    rows: list[dict[str, float | int | str]] = []
    for label, method, dimension, summary, record_summary in specifications:
        segment_row = _select_row(summary, method, dimension)
        record_row = _select_row(
            record_summary, method, dimension, "record_mean_probability"
        )
        rows.append(
            {
                "representation": label,
                "method": method,
                "dimension": dimension,
                "segment_balanced_accuracy_mean": segment_row[
                    "utility_balanced_accuracy_mean"
                ],
                "segment_balanced_accuracy_std": segment_row[
                    "utility_balanced_accuracy_std"
                ],
                "record_balanced_accuracy_mean": record_row[
                    "balanced_accuracy_mean"
                ],
                "record_balanced_accuracy_std": record_row[
                    "balanced_accuracy_std"
                ],
                "pairwise_linkability_roc_auc_mean": segment_row[
                    "linkability_roc_auc_mean"
                ],
                "pairwise_linkability_roc_auc_std": segment_row[
                    "linkability_roc_auc_std"
                ],
                "n_seeds": int(segment_row["n_seeds"]),
            }
        )

    figure_data = pd.DataFrame(rows)

    # Protect the figure against accidentally drifting from the rounded values in
    # the manuscript's controlled main-comparison table.
    expected = {
        "Raw handcrafted": (0.8597, 0.9986),
        "PCA, 8-D": (0.6860, 0.9651),
        "Random projection, 8-D": (0.6773, 0.8137),
        "Supervised encoder, 8-D": (0.8978, 0.8825),
        "Encoder + Gaussian, C=2": (0.8945, 0.8074),
        "Encoder + Gaussian, C=4": (0.8872, 0.7233),
    }
    for row in figure_data.itertuples(index=False):
        expected_record_ba, expected_link_auc = expected[row.representation]
        if not np.isclose(
            row.record_balanced_accuracy_mean, expected_record_ba, atol=5e-5
        ) or not np.isclose(
            row.pairwise_linkability_roc_auc_mean,
            expected_link_auc,
            atol=5e-5,
        ):
            raise ValueError(
                f"Figure data for {row.representation!r} no longer match Table I."
            )

    return figure_data


def plot_main_tradeoff(figure_data: pd.DataFrame) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7.2,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
        }
    )

    order = [
        "Raw handcrafted",
        "PCA, 8-D",
        "Random projection, 8-D",
        "Supervised encoder, 8-D",
        "Encoder + Gaussian, C=2",
        "Encoder + Gaussian, C=4",
    ]
    labels = [
        "Handcrafted",
        "PCA",
        "Random projection",
        "Encoder",
        "Gaussian, $C=2$",
        "Gaussian, $C=4$",
    ]
    styles = {
        "Raw handcrafted": ("#4D4D4D", "D", 29),
        "PCA, 8-D": ("#D55E00", "s", 29),
        "Random projection, 8-D": ("#D55E00", "^", 32),
        "Supervised encoder, 8-D": ("#0072B2", "o", 34),
        "Encoder + Gaussian, C=2": ("#009E73", "o", 34),
        "Encoder + Gaussian, C=4": ("#009E73", "P", 37),
    }
    plotted = figure_data.set_index("representation").loc[order]
    positions = np.arange(len(order))

    fig, (utility_ax, link_ax) = plt.subplots(
        1,
        2,
        figsize=(3.5, 2.45),
        sharey=True,
        gridspec_kw={"wspace": 0.18},
    )

    for axis in (utility_ax, link_ax):
        axis.set_axisbelow(True)
        axis.grid(axis="x", color="#D9D9D9", linewidth=0.55, alpha=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_visible(False)
        axis.tick_params(axis="y", length=0)

    for position, (representation, row) in enumerate(plotted.iterrows()):
        color, marker, size = styles[representation]
        panels = [
            (
                utility_ax,
                row["record_balanced_accuracy_mean"],
                row["record_balanced_accuracy_std"],
            ),
            (
                link_ax,
                row["pairwise_linkability_roc_auc_mean"],
                row["pairwise_linkability_roc_auc_std"],
            ),
        ]
        for axis, value, error in panels:
            axis.errorbar(
                value,
                position,
                xerr=error,
                fmt="none",
                ecolor=color,
                elinewidth=0.85,
                capsize=2.0,
                capthick=0.85,
                alpha=0.85,
                zorder=2,
            )
            axis.scatter(
                value,
                position,
                s=size,
                color=color,
                marker=marker,
                edgecolor="white",
                linewidth=0.55,
                zorder=3,
            )

    utility_ax.set_yticks(positions, labels)
    utility_ax.tick_params(axis="y", labelsize=6.5, pad=2)
    utility_ax.invert_yaxis()
    link_ax.tick_params(axis="y", labelleft=False)

    utility_ax.set_title("(a) Utility", fontsize=7.4, pad=3)
    utility_ax.set_xlabel("Record BA")
    utility_ax.set_xlim(0.64, 0.915)
    utility_ax.set_xticks([0.65, 0.75, 0.85])

    link_ax.set_title("(b) Linkability", fontsize=7.4, pad=3)
    link_ax.set_xlabel("Link ROC-AUC")
    link_ax.set_xlim(0.69, 1.015)
    link_ax.set_xticks([0.70, 0.80, 0.90, 1.00])

    fig.subplots_adjust(left=0.43, right=0.99, bottom=0.21, top=0.91)
    return fig


def save_outputs(figure_data: pd.DataFrame, figure: plt.Figure) -> list[Path]:
    OUTPUTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    data_path = OUTPUTS_TABLES_DIR / "main_utility_linkability_tradeoff_figure_data.csv"
    figure_data.to_csv(data_path, index=False)

    output_paths = [
        OUTPUTS_FIGURES_DIR / "main_utility_linkability_tradeoff_clean.png",
        PUBLIC_FIGURES_DIR / "main_utility_linkability_tradeoff.png",
    ]
    for path in output_paths:
        save_options = {"bbox_inches": "tight", "pad_inches": 0.02}
        if path.suffix == ".png":
            save_options["dpi"] = 600
        figure.savefig(path, **save_options)
    plt.close(figure)
    return [data_path, *output_paths]


def main() -> None:
    figure_data = build_figure_data()
    figure = plot_main_tradeoff(figure_data)
    for path in save_outputs(figure_data, figure):
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
