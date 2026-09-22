from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from run_attribute_inference_attack import build_true_view
from run_reconstruction_attack import max_lag_correlation, reconstruction_metrics, rowwise_correlations
from summarize_attack_extensions import metric_effect


class AttributeInferenceExtensionTests(unittest.TestCase):
    def test_true_view_uses_only_complete_records(self) -> None:
        frame = pd.DataFrame(
            {
                "patient_id": ["A", "A", "B"],
                "segment_ref": [0, 4, 0],
                "utility_label": ["Atrial", "Atrial", "Non-Atrial"],
                "feature": [1.0, 3.0, 10.0],
            }
        )
        view = build_true_view(frame, feature_columns=["feature"], refs=[0, 4])
        self.assertEqual(view["patient_id"].tolist(), ["A"])
        self.assertAlmostEqual(float(view.loc[0, "feature"]), 2.0)

    def test_positive_effect_means_improved_privacy_attack(self) -> None:
        y_true = np.asarray([0, 0, 1, 1])
        baseline = np.asarray([0, 1, 0, 1])
        improved = np.asarray([0, 0, 1, 1])
        self.assertGreater(metric_effect("sex", y_true, baseline, improved), 0.0)

        ages = np.asarray([20.0, 40.0, 60.0])
        baseline_age = np.asarray([30.0, 50.0, 70.0])
        improved_age = np.asarray([22.0, 42.0, 62.0])
        self.assertGreater(metric_effect("age", ages, baseline_age, improved_age), 0.0)


class ReconstructionExtensionTests(unittest.TestCase):
    def test_rowwise_correlation_identical_and_inverted(self) -> None:
        signal = np.asarray([[0.0, 1.0, 0.0, -1.0]])
        self.assertAlmostEqual(float(rowwise_correlations(signal, signal)[0]), 1.0)
        self.assertAlmostEqual(float(rowwise_correlations(signal, -signal)[0]), -1.0)

    def test_max_lag_recovers_shifted_shape(self) -> None:
        signal = np.zeros((1, 20), dtype=float)
        signal[0, 5] = 1.0
        shifted = np.zeros((1, 20), dtype=float)
        shifted[0, 7] = 1.0
        direct = float(rowwise_correlations(signal, shifted)[0])
        lagged = float(max_lag_correlation(signal, shifted, max_lag=3)[0])
        self.assertGreater(lagged, direct)
        self.assertGreater(lagged, 0.99)

    def test_reconstruction_metrics_are_exact_for_identical_signal(self) -> None:
        # Two records, 12 leads, four samples per lead.
        waveforms = np.arange(2 * 12 * 4, dtype=float).reshape(2, -1)
        summary, per_record = reconstruction_metrics(
            patient_ids=np.asarray(["A", "B"]),
            y_true_flat=waveforms,
            y_pred_flat=waveforms.copy(),
            n_leads=12,
            target_fs=100,
            max_lag_ms=20,
        )
        self.assertAlmostEqual(summary["rmse_mean"], 0.0)
        self.assertAlmostEqual(summary["prd_mean"], 0.0)
        self.assertTrue(np.allclose(per_record["lead_ii_correlation"], 1.0))


if __name__ == "__main__":
    unittest.main()
