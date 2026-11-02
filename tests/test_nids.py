"""Tests for NIDS models, synthetic traffic, and calibration."""

import unittest
import numpy as np
import torch

from src.nids.synthetic_data import SyntheticTrafficGenerator, create_benchmark_splits
from src.nids.contrastive_model import MultimodalNIDS, augment_flow
from src.nids.calibration import TemperatureScaler, compute_ece
from src.nids.baselines import BaselineRandomForest, mcnemar_test


class TestNIDS(unittest.TestCase):

    def test_synthetic_traffic_generator(self):
        gen = SyntheticTrafficGenerator()
        self.assertEqual(len(gen.class_names), 10)
        flows = gen.generate_flows("DDoS", 50)
        self.assertEqual(flows.shape, (50, 42))
        self.assertTrue(np.all(flows >= 0.0))

    def test_benchmark_splits(self):
        splits = create_benchmark_splits(dataset_name="CSE-CIC-IDS2018", total_samples=500)
        self.assertIn("train", splits)
        self.assertIn("val", splits)
        self.assertIn("test_closed", splits)
        self.assertIn("test_zero_shot", splits)
        self.assertIn("Infiltration", splits["held_out_classes"])

    def test_contrastive_model_forward_and_loss(self):
        model = MultimodalNIDS(input_dim=42, hidden_dim=64, embedding_dim=32, num_classes=10)
        x = torch.randn(8, 42)
        labels = torch.randint(0, 10, (8,))

        z, logits = model(x)
        self.assertEqual(z.shape, (8, 32))
        self.assertEqual(logits.shape, (8, 10))
        norm = torch.norm(z, p=2, dim=-1)
        self.assertTrue(torch.allclose(norm, torch.ones_like(norm), atol=1e-5))

        loss, details = model.compute_loss(x, labels)
        self.assertGreater(loss.item(), 0.0)
        self.assertIn("loss_traffic", details)
        self.assertIn("loss_multi", details)

    def test_temperature_scaling_and_ece(self):
        scaler = TemperatureScaler(initial_temp=1.0)
        logits = torch.randn(100, 5)
        labels = torch.randint(0, 5, (100,))

        best_tau = scaler.fit(logits, labels, max_iter=20)
        self.assertGreater(best_tau, 0.0)

        probs = torch.softmax(logits, dim=-1).numpy()
        ece, mce, _ = compute_ece(probs, labels.numpy(), n_bins=10)
        self.assertTrue(0.0 <= ece <= 1.0)
        self.assertTrue(0.0 <= mce <= 1.0)

    def test_mcnemar_test(self):
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1])
        y_pred_a = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1])
        y_pred_b = np.array([0, 0, 0, 0, 1, 0, 1, 0, 1, 1])
        chi2, p_val, counts = mcnemar_test(y_true, y_pred_a, y_pred_b)
        self.assertEqual(counts["b"], 2)
        self.assertEqual(counts["c"], 0)
        self.assertGreaterEqual(chi2, 0.0)


if __name__ == "__main__":
    unittest.main()
