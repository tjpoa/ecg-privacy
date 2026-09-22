from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

try:
    from config import OUTPUTS_TABLES_DIR
except ImportError:  # pragma: no cover - package import fallback
    from .config import OUTPUTS_TABLES_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate segment-level utility predictions into retained-record predictions."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--run-name", default="record_level_utility")
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_TABLES_DIR)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def evaluate_predictions(frame: pd.DataFrame, threshold: float) -> dict[str, float | int | str]:
    y_true = frame["y_true"].astype(int).to_numpy()
    y_score = frame["y_score"].astype(float).to_numpy()
    y_pred = (y_score >= threshold).astype(int)
    return {
        "n_predictions": int(len(frame)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(args.predictions)
    required = {"seed", "method", "dimension", "patient_id", "y_true", "y_score"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")

    rows: list[dict[str, object]] = []
    group_columns = ["seed", "method", "dimension"]
    for keys, segment_frame in predictions.groupby(group_columns, sort=True):
        seed, method, dimension = keys
        segment_metrics = evaluate_predictions(segment_frame, args.threshold)
        rows.append(
            {
                "seed": int(seed),
                "method": str(method),
                "dimension": int(dimension),
                "evaluation_level": "segment",
                **segment_metrics,
            }
        )

        if segment_frame.groupby("patient_id")["y_true"].nunique().max() != 1:
            raise AssertionError("A retained record has inconsistent utility labels.")
        record_frame = (
            segment_frame.groupby("patient_id", as_index=False)
            .agg(y_true=("y_true", "first"), y_score=("y_score", "mean"), n_segments=("y_score", "size"))
        )
        record_metrics = evaluate_predictions(record_frame, args.threshold)
        rows.append(
            {
                "seed": int(seed),
                "method": str(method),
                "dimension": int(dimension),
                "evaluation_level": "record_mean_probability",
                **record_metrics,
            }
        )

    metrics = pd.DataFrame(rows)
    metric_columns = ["n_predictions", "f1", "balanced_accuracy", "roc_auc", "pr_auc"]
    summary = metrics.groupby(
        ["method", "dimension", "evaluation_level"], as_index=False
    )[metric_columns].agg(["mean", "std"])
    summary.columns = [
        "_".join(part for part in column if part) if isinstance(column, tuple) else column
        for column in summary.columns
    ]

    metrics_path = args.output_dir / f"{args.run_name}_metrics.csv"
    summary_path = args.output_dir / f"{args.run_name}_summary.csv"
    protocol_path = args.output_dir / f"{args.run_name}_protocol.json"
    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    protocol_path.write_text(
        json.dumps(
            {
                "source_predictions": str(args.predictions),
                "threshold": args.threshold,
                "aggregation": "Arithmetic mean of segment positive-class probabilities within each retained record.",
                "outputs": {"metrics": str(metrics_path), "summary": str(summary_path)},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print("Saved metrics:", metrics_path)
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()
