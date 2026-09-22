from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modeling import (
    assert_disjoint_group_frames,
    evaluate_utility_model_on_split,
    split_segments_by_patient_stratified,
)


def synthetic_segments() -> pd.DataFrame:
    rows = []
    for record_index in range(10):
        label = "Atrial" if record_index % 2 == 0 else "NonAtrial"
        for segment_index in range(3):
            rows.append({
                "patient_id": f"R{record_index}",
                "segment_ref": segment_index,
                "utility_label": label,
                "feature_a": record_index + segment_index * 0.1,
                "feature_b": (record_index % 2) + segment_index * 0.2,
            })
    return pd.DataFrame(rows)


class GroupSplitTests(unittest.TestCase):
    def test_stratified_split_is_group_disjoint(self) -> None:
        train_df, test_df = split_segments_by_patient_stratified(
            synthetic_segments(),
            random_state=42,
        )
        assert_disjoint_group_frames(train_df, test_df)
        self.assertTrue(set(train_df["patient_id"]).isdisjoint(set(test_df["patient_id"])))

    def test_overlap_is_rejected(self) -> None:
        frame = synthetic_segments()
        with self.assertRaises(AssertionError):
            assert_disjoint_group_frames(frame.iloc[:2], frame.iloc[1:3])

    def test_utility_evaluation_returns_predictions(self) -> None:
        train_df, test_df = split_segments_by_patient_stratified(
            synthetic_segments(),
            random_state=42,
        )
        output = evaluate_utility_model_on_split(train_df, test_df)
        predictions = output["predictions_df"]
        self.assertEqual(len(predictions), len(test_df))
        self.assertTrue({"y_true", "y_pred", "y_score"}.issubset(predictions.columns))


if __name__ == "__main__":
    unittest.main()
