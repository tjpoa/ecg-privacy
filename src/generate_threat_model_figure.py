from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

try:
    from config import OUTPUTS_FIGURES_DIR, PROJECT_ROOT
except ImportError:  # pragma: no cover - package import fallback
    from .config import OUTPUTS_FIGURES_DIR, PROJECT_ROOT


MANUSCRIPT_FIGURES_DIR = (
    PROJECT_ROOT / "manuscript" / "overleaf_jbhi_submission" / "figures"
)


def _rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    edge: str,
    fill: str,
    *,
    fontsize: float = 8.2,
    text_color: str = "#222222",
    weight: str = "normal",
) -> None:
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0.9,
        edgecolor=edge,
        facecolor=fill,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
        weight=weight,
        linespacing=1.05,
    )


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=1.15,
        color=color,
        shrinkA=1,
        shrinkB=1,
        clip_on=False,
    )
    ax.add_patch(arrow)


def build_figure() -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    blue = "#2B6F9E"
    blue_fill = "#EAF3F9"
    blue_lane = "#F6F9FC"
    green = "#16876A"
    green_fill = "#EAF6F1"
    green_lane = "#F5FAF8"

    fig, ax = plt.subplots(figsize=(7.12, 1.58))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Subtle lanes make the independence of the two attack paths immediately
    # visible without adding a separate legend.
    ax.add_patch(
        FancyBboxPatch(
            (0.006, 0.535),
            0.988,
            0.42,
            boxstyle="round,pad=0.004,rounding_size=0.018",
            linewidth=0,
            facecolor=blue_lane,
            transform=ax.transAxes,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (0.006, 0.045),
            0.988,
            0.42,
            boxstyle="round,pad=0.004,rounding_size=0.018",
            linewidth=0,
            facecolor=green_lane,
            transform=ax.transAxes,
        )
    )

    _rounded_box(
        ax,
        0.018,
        0.615,
        0.105,
        0.26,
        "GROUPING\nAXIS",
        blue,
        blue,
        fontsize=7.3,
        text_color="white",
        weight="bold",
    )
    _rounded_box(
        ax,
        0.018,
        0.125,
        0.105,
        0.26,
        "INVERSION\nAXIS",
        green,
        green,
        fontsize=7.3,
        text_color="white",
        weight="bold",
    )

    top_y = 0.645
    top_h = 0.20
    top_boxes = [
        (0.15, 0.16, "Released segment\nrepresentations"),
        (0.37, 0.16, "Pairwise same-record\nscorer"),
        (0.59, 0.16, "Candidate ranking\nand grouping"),
        (0.81, 0.17, "Attribute aggregation\n(sex and age)"),
    ]
    for x, width, label in top_boxes:
        _rounded_box(ax, x, top_y, width, top_h, label, blue, blue_fill)
    for left, right in zip(top_boxes[:-1], top_boxes[1:]):
        _arrow(
            ax,
            (left[0] + left[1], top_y + top_h / 2),
            (right[0], top_y + top_h / 2),
            blue,
        )

    bottom_y = 0.185
    bottom_h = 0.20
    bottom_boxes = [
        (0.18, 0.20, "One released segment\nrepresentation"),
        (0.47, 0.17, "Waveform decoder"),
        (0.73, 0.22, "Reconstructed 12-lead\nwaveform"),
    ]
    for x, width, label in bottom_boxes:
        _rounded_box(ax, x, bottom_y, width, bottom_h, label, green, green_fill)
    for left, right in zip(bottom_boxes[:-1], bottom_boxes[1:]):
        _arrow(
            ax,
            (left[0] + left[1], bottom_y + bottom_h / 2),
            (right[0], bottom_y + bottom_h / 2),
            green,
        )

    ax.text(
        0.565,
        0.085,
        "Direct path: no pairing, candidate ranking, or grouping",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.2,
        color="#355E52",
        style="italic",
    )

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig


def save_outputs(figure: plt.Figure) -> list[Path]:
    OUTPUTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        OUTPUTS_FIGURES_DIR / "threat_model_attack_axes_clean.png",
        OUTPUTS_FIGURES_DIR / "threat_model_attack_axes_clean.pdf",
        MANUSCRIPT_FIGURES_DIR / "threat_model_attack_axes.png",
        MANUSCRIPT_FIGURES_DIR / "threat_model_attack_axes.pdf",
    ]
    for path in paths:
        options: dict[str, float | str] = {
            "bbox_inches": "tight",
            "pad_inches": 0.02,
            "facecolor": "white",
        }
        if path.suffix == ".png":
            options["dpi"] = 600
        figure.savefig(path, **options)
    plt.close(figure)
    return paths


def main() -> None:
    for path in save_outputs(build_figure()):
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
