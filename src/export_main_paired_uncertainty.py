"""Export paired uncertainty for the article's main representation contrasts.

The controlled experiments reuse the same held-out records and sampled pairs
within each seed.  This script first forms within-seed metric differences and
then reports a Student-t interval across the three paired seed differences.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
SEEDS = [42, 123, 456]
CONFIDENCE = 0.95


CONTRASTS = [
    {
        "contrast": "PCA 8-D minus raw handcrafted",
        "short_label": "PCA, 8-D - Raw",
        "source": "dimension",
        "candidate": "pca",
        "candidate_dim": 8,
        "reference": "identity",
        "reference_dim": 208,
    },
    {
        "contrast": "Random projection 8-D minus raw handcrafted",
        "short_label": "Random projection, 8-D - Raw",
        "source": "dimension",
        "candidate": "random_projection",
        "candidate_dim": 8,
        "reference": "identity",
        "reference_dim": 208,
    },
    {
        "contrast": "Supervised encoder 8-D minus raw handcrafted",
        "short_label": "Encoder, 8-D - Raw",
        "source": "dimension",
        "candidate": "supervised_encoder",
        "candidate_dim": 8,
        "reference": "identity",
        "reference_dim": 208,
    },
    {
        "contrast": "Gaussian C=2 minus supervised encoder 8-D",
        "short_label": "Gaussian C=2 - Encoder",
        "source": "controlled",
        "candidate": "encoder_gaussian_c2",
        "candidate_dim": 8,
        "reference": "supervised_encoder",
        "reference_dim": 8,
    },
    {
        "contrast": "Gaussian C=4 minus supervised encoder 8-D",
        "short_label": "Gaussian C=4 - Encoder",
        "source": "controlled",
        "candidate": "encoder_gaussian_c4",
        "candidate_dim": 8,
        "reference": "supervised_encoder",
        "reference_dim": 8,
    },
]


def read_csv(name: str) -> pd.DataFrame:
    path = TABLES / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def select_rows(frame: pd.DataFrame, method: str, dimension: int) -> pd.DataFrame:
    selected = frame[frame["method"].eq(method) & frame["dimension"].eq(dimension)].copy()
    if sorted(selected["seed"].astype(int).tolist()) != SEEDS:
        raise ValueError(f"Expected one row per seed for {method}, {dimension}-D")
    return selected.set_index("seed").sort_index()


def t_interval(values: pd.Series) -> dict[str, float | int]:
    array = values.to_numpy(dtype=float)
    n = len(array)
    if n != len(SEEDS):
        raise ValueError(f"Expected {len(SEEDS)} paired differences, got {n}")
    mean = float(np.mean(array))
    sd = float(np.std(array, ddof=1))
    se = sd / np.sqrt(n)
    critical = float(t.ppf(0.5 + CONFIDENCE / 2.0, df=n - 1))
    return {
        "n_paired_seeds": n,
        "df": n - 1,
        "paired_mean_difference": mean,
        "paired_sd": sd,
        "paired_se": se,
        "ci_level": CONFIDENCE,
        "ci_low": mean - critical * se,
        "ci_high": mean + critical * se,
    }


def main() -> None:
    sources = {
        "dimension": {
            "metrics": read_csv("representation_dimension_submission_matched_metrics.csv"),
            "records": read_csv("representation_dimension_submission_matched_record_level_metrics.csv"),
        },
        "controlled": {
            "metrics": read_csv("article_controlled_representation_multiseed_metrics.csv"),
            "records": read_csv("article_controlled_record_level_utility_metrics.csv"),
        },
    }
    for source in sources.values():
        source["records"] = source["records"][
            source["records"]["evaluation_level"].eq("record_mean_probability")
        ].copy()

    seed_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    metric_specs = [
        ("segment_balanced_accuracy", "utility_balanced_accuracy", "metrics"),
        ("record_balanced_accuracy", "balanced_accuracy", "records"),
        ("pairwise_linkability_roc_auc", "linkability_roc_auc", "metrics"),
    ]

    for order, contrast in enumerate(CONTRASTS, start=1):
        source = sources[str(contrast["source"])]
        selected_frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
        for _, _, frame_key in metric_specs:
            if frame_key in selected_frames:
                continue
            frame = source[frame_key]
            candidate = select_rows(
                frame, str(contrast["candidate"]), int(contrast["candidate_dim"])
            )
            reference = select_rows(
                frame, str(contrast["reference"]), int(contrast["reference_dim"])
            )
            selected_frames[frame_key] = (candidate, reference)

        metric_differences: dict[str, pd.Series] = {}
        for metric_name, column, frame_key in metric_specs:
            candidate, reference = selected_frames[frame_key]
            if frame_key == "metrics":
                if not np.array_equal(candidate["link_test_pairs"], reference["link_test_pairs"]):
                    raise ValueError(f"Pair-count mismatch for {contrast['contrast']}")
            else:
                if not np.array_equal(candidate["n_predictions"], reference["n_predictions"]):
                    raise ValueError(f"Record-count mismatch for {contrast['contrast']}")
            difference = candidate[column].astype(float) - reference[column].astype(float)
            metric_differences[metric_name] = difference
            interval = t_interval(difference)
            summary_rows.append(
                {
                    "contrast_order": order,
                    "contrast": contrast["contrast"],
                    "short_label": contrast["short_label"],
                    "metric": metric_name,
                    **interval,
                }
            )

        for seed in SEEDS:
            row: dict[str, object] = {
                "contrast_order": order,
                "contrast": contrast["contrast"],
                "short_label": contrast["short_label"],
                "source_protocol": contrast["source"],
                "seed": seed,
                "candidate_method": contrast["candidate"],
                "candidate_dimension": contrast["candidate_dim"],
                "reference_method": contrast["reference"],
                "reference_dimension": contrast["reference_dim"],
            }
            for metric_name, difference in metric_differences.items():
                row[f"delta_{metric_name}"] = float(difference.loc[seed])
            seed_rows.append(row)

    seed_frame = pd.DataFrame(seed_rows)
    summary_frame = pd.DataFrame(summary_rows)
    seed_path = TABLES / "main_paired_contrast_seed_differences.csv"
    summary_path = TABLES / "main_paired_uncertainty_summary.csv"
    protocol_path = TABLES / "main_paired_uncertainty_protocol.json"
    seed_frame.to_csv(seed_path, index=False)
    summary_frame.to_csv(summary_path, index=False)

    protocol = {
        "estimand": "candidate metric minus reference metric within the same seed",
        "pairing": (
            "Every contrast pairs metrics computed on the same held-out retained records. "
            "Segment and record utility use identical test predictions, and linkability uses "
            "the identical sampled test-pair table within each source run."
        ),
        "interval": (
            "Two-sided 95% Student-t interval over the three paired seed-level differences "
            "(n=3, df=2)."
        ),
        "interpretation": (
            "Negative balanced-accuracy differences indicate utility loss. Negative ROC-AUC "
            "differences indicate weaker linkability. With only three seeds, intervals are "
            "descriptive robustness summaries rather than population-level inferential guarantees."
        ),
        "multiplicity_adjustment": "none; five prespecified descriptive contrasts",
        "seeds": SEEDS,
        "confidence_level": CONFIDENCE,
        "source_tables": {
            "dimension_metrics": "representation_dimension_submission_matched_metrics.csv",
            "dimension_record_metrics": "representation_dimension_submission_matched_record_level_metrics.csv",
            "controlled_metrics": "article_controlled_representation_multiseed_metrics.csv",
            "controlled_record_metrics": "article_controlled_record_level_utility_metrics.csv",
        },
        "outputs": [seed_path.name, summary_path.name],
    }
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"Saved {seed_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {protocol_path}")


if __name__ == "__main__":
    main()
