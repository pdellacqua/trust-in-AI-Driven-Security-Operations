"""Tests for simulation engine, stimulus stream generator, and analyst agent."""

import unittest
import numpy as np

from src.simulation.alert_stream import AlertStreamGenerator
from src.simulation.analyst_agent import SimulatedAnalystAgent
from src.simulation.closed_loop import ClosedLoopSimulation
from src.evaluation.metrics import compute_condition_metrics


class TestSimulation(unittest.TestCase):

    def test_alert_stream_structure(self):
        gen = AlertStreamGenerator(seed=42)
        block = gen.generate_block(block_id=1)
        self.assertEqual(len(block), 20)

        benign_count = sum(1 for a in block if a.true_class == "Benign")
        self.assertEqual(benign_count, 10)

        perts = [a for a in block if a.is_perturbation]
        self.assertEqual(len(perts), 2)

    def test_closed_loop_participant_session(self):
        sim = ClosedLoopSimulation()
        session = sim.run_participant_session(participant_id=1)

        self.assertEqual(len(session["trial_results"]), 60)
        self.assertEqual(len(session["block_summaries"]), 3)
        self.assertEqual(len(session["condition_order"]), 3)

        trials = session["trial_results"]
        metrics = compute_condition_metrics(trials, session["block_summaries"])
        self.assertTrue(0.0 <= metrics["accuracy"] <= 1.0)
        self.assertTrue(0.0 <= metrics["mace"] <= 1.0)
        self.assertGreater(metrics["mean_time"], 0.0)


if __name__ == "__main__":
    unittest.main()
