"""
Adaptive Explanation Cost Function and Asymmetric Trust Calibration Penalties.
Implements Equations (11)-(13), (33)-(36), and (39) of the paper.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np


class ControllerCostEvaluator:
    """
    Evaluates candidates a in A = {E0, E1, E2, E3} according to operational risks,
    miscalibration magnitude, cognitive load, and intrinsic interaction costs.
    """

    def __init__(
        self,
        w1: float = 1.00,
        w2: float = 0.45,
        w3: float = 0.30,
        w4: float = 0.60,
        w5: float = 1.20,
        w6: float = 0.80,
        kappa_intensity: Optional[List[float]] = None,
        lambda_complexity: Optional[List[float]] = None,
        kappa_high: float = 0.70,
        kappa_low: float = 0.40,
    ):
        self.w1 = w1  # Miscalibration penalty weight
        self.w2 = w2  # Cognitive load weight
        self.w3 = w3  # Explanation complexity cost weight
        self.w4 = w4  # Task alignment weight
        self.w5 = w5  # Asymmetric overtrust penalty weight (w5 > w6)
        self.w6 = w6  # Asymmetric undertrust penalty weight

        # Action parameters for E0, E1, E2, E3
        self.kappa = np.array(kappa_intensity or [0.1, 0.4, 0.7, 1.0], dtype=np.float64)
        self.lambdas = np.array(lambda_complexity or [0.0, 0.2, 0.5, 0.9], dtype=np.float64)

        self.kappa_high = kappa_high
        self.kappa_low = kappa_low

    def intrinsic_cost(self, action_idx: int) -> float:
        """F(a) = lambda_a * kappa(a)^2."""
        return float(self.lambdas[action_idx] * (self.kappa[action_idx] ** 2))

    def task_risk(self, action_idx: int, difficulty: float) -> float:
        """R_task(a, d_t) = [d_t - kappa(a)]^2."""
        return float((difficulty - self.kappa[action_idx]) ** 2)

    def overtrust_penalty(self, action_idx: int) -> float:
        """
        R_over(a) = max(0, kappa_high - kappa(a))^2.
        Penalizes shallow explanations when operator is overtrusting,
        steering optimization toward deep explanations exposing uncertainty.
        """
        diff = max(0.0, self.kappa_high - self.kappa[action_idx])
        return float(diff ** 2)

    def undertrust_penalty(self, action_idx: int) -> float:
        """
        R_under(a) = max(0, kappa_low - kappa(a))^2.
        Penalizes overly sparse explanations under undertrust,
        ensuring sufficient corroborating evidence to prevent unwarranted rejection.
        """
        diff = max(0.0, self.kappa_low - self.kappa[action_idx])
        return float(diff ** 2)

    def evaluate_action_cost(
        self,
        action_idx: int,
        inferred_trust: float,
        cognitive_load: float,
        reliability: float,
        difficulty: float,
    ) -> float:
        """
        Computes full cost J_t(a):
        J_t(a) = w1 * (T_hat_t - r_t)^2
               + w2 * C_hat_t
               + w3 * F(a)
               + w4 * R_task(a, d_t)
               + w5 * R_over(a) * max(0, TCE_t)
               + w6 * R_under(a) * max(0, -TCE_t)
        """
        tce = inferred_trust - reliability
        miscal_term = self.w1 * (tce ** 2)
        load_term = self.w2 * cognitive_load
        intrinsic_term = self.w3 * self.intrinsic_cost(action_idx)
        task_term = self.w4 * self.task_risk(action_idx, difficulty)
        over_term = self.w5 * self.overtrust_penalty(action_idx) * max(0.0, tce)
        under_term = self.w6 * self.undertrust_penalty(action_idx) * max(0.0, -tce)

        total_cost = miscal_term + load_term + intrinsic_term + task_term + over_term + under_term
        return float(total_cost)

    def evaluate_trust_agnostic_cost(
        self,
        action_idx: int,
        cognitive_load: float,
        reliability: float,
        difficulty: float,
        w_r: float = 1.00,
        w_c: float = 0.45,
    ) -> float:
        """
        Evaluates Trust-Agnostic baseline cost J_t^{TA}(a):
        J_t^{TA}(a) = w3 * F(a)
                    + w4 * [d_t - kappa(a)]^2
                    + w_r * (1 - r_t) * [kappa_high - kappa(a)]^2
                    + w_c * C_hat_t * kappa(a)
        Deliberately excludes inferred trust T_hat_t and TCE_t.
        """
        f_cost = self.w3 * self.intrinsic_cost(action_idx)
        task_term = self.w4 * ((difficulty - self.kappa[action_idx]) ** 2)
        rel_term = w_r * (1.0 - reliability) * ((self.kappa_high - self.kappa[action_idx]) ** 2)
        load_term = w_c * cognitive_load * self.kappa[action_idx]

        return float(f_cost + task_term + rel_term + load_term)
