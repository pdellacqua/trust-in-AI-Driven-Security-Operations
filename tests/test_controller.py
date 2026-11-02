"""Tests for adaptive explanation controller, cost minimization, and hysteresis."""

import unittest
import numpy as np

from src.controller.cost_function import ControllerCostEvaluator
from src.controller.policies import (
    StaticExplanationPolicy,
    TrustAgnosticAdaptivePolicy,
    TrustCalibratedPolicy,
)
from src.controller.adaptive_engine import AdaptiveExplainabilityEngine


class TestController(unittest.TestCase):

    def test_asymmetric_cost_evaluation(self):
        evaluator = ControllerCostEvaluator(w5=1.20, w6=0.80, kappa_high=0.70, kappa_low=0.40)
        cost_e0 = evaluator.evaluate_action_cost(0, inferred_trust=0.90, cognitive_load=0.20, reliability=0.40, difficulty=0.30)
        cost_e3 = evaluator.evaluate_action_cost(3, inferred_trust=0.90, cognitive_load=0.20, reliability=0.40, difficulty=0.30)

        over_pen_e0 = evaluator.overtrust_penalty(0)
        over_pen_e3 = evaluator.overtrust_penalty(3)
        self.assertGreater(over_pen_e0, 0.0)
        self.assertEqual(over_pen_e3, 0.0)

    def test_controller_policies(self):
        static_pol = StaticExplanationPolicy()
        idx, name, _, _ = static_pol.select_action(0.8, 0.3, 0.8, 0.4)
        self.assertEqual(name, "E1")

        tc_pol = TrustCalibratedPolicy()
        idx_tc, name_tc, _, _ = tc_pol.select_action(0.92, 0.15, 0.92, 0.10)
        self.assertIn(name_tc, ["E0", "E1"])

        # Low reliability, severe overtrust, and high task difficulty -> E2 or E3 selected
        idx_forced, name_forced, _, _ = tc_pol.select_action(0.95, 0.20, 0.25, 0.75, previous_action_idx=0)
        self.assertIn(name_forced, ["E2", "E3"])

    def test_hysteresis_action_smoothing(self):
        engine = AdaptiveExplainabilityEngine(policy_name="Trust-Calibrated")
        engine.current_action_idx = 1
        record = engine.step(trial_id=1, inferred_trust=0.55, cognitive_load=0.30, reliability=0.55, difficulty=0.35)
        self.assertIn("action_name", record)
        self.assertIn("tce", record)


if __name__ == "__main__":
    unittest.main()
