"""
Explanation Adaptation Policies for Closed-Loop Threat Triage.
Implements:
- Static Policy (pi_S): Fixed intermediate explanation E1
- Trust-Agnostic Adaptive Policy (pi_TA): Context and workload driven without latent trust
- Trust-Calibrated Adaptive Policy (pi_TC): Full asymmetric closed-loop optimization
- Ablation policies: Reliability-Only, Trust-Only, Workload-Only
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import numpy as np
from .cost_function import ControllerCostEvaluator


class BaseExplanationPolicy(ABC):
    """Abstract base class for interface explanation policies."""

    @abstractmethod
    def select_action(
        self,
        inferred_trust: float,
        cognitive_load: float,
        reliability: float,
        difficulty: float,
        previous_action_idx: int = 1,
    ) -> Tuple[int, str, float, Dict[str, float]]:
        """
        Returns:
            action_idx: index in [0, 1, 2, 3]
            action_name: "E0", "E1", "E2", or "E3"
            action_cost: scalar cost of selected action
            all_costs: dict of candidate costs
        """
        pass


class StaticExplanationPolicy(BaseExplanationPolicy):
    """Static explanation policy fixing interface permanently to E1."""

    def __init__(self, fixed_level_idx: int = 1):
        self.fixed_idx = fixed_level_idx
        self.action_names = ["E0", "E1", "E2", "E3"]

    def select_action(
        self,
        inferred_trust: float,
        cognitive_load: float,
        reliability: float,
        difficulty: float,
        previous_action_idx: int = 1,
    ) -> Tuple[int, str, float, Dict[str, float]]:
        action_name = self.action_names[self.fixed_idx]
        return self.fixed_idx, action_name, 0.0, {action_name: 0.0}


class TrustAgnosticAdaptivePolicy(BaseExplanationPolicy):
    """
    Trust-Agnostic Adaptive Policy (pi_TA).
    Minimizes J_t^{TA}(a) based on model reliability r_t, task difficulty d_t,
    and cognitive strain C_hat_t without latent trust modeling.
    """

    def __init__(
        self,
        cost_evaluator: Optional[ControllerCostEvaluator] = None,
        delta_switch: float = 0.15,
        w_r: float = 1.00,
        w_c: float = 0.45,
    ):
        self.evaluator = cost_evaluator or ControllerCostEvaluator()
        self.delta_switch = delta_switch
        self.w_r = w_r
        self.w_c = w_c
        self.action_names = ["E0", "E1", "E2", "E3"]

    def select_action(
        self,
        inferred_trust: float,
        cognitive_load: float,
        reliability: float,
        difficulty: float,
        previous_action_idx: int = 1,
    ) -> Tuple[int, str, float, Dict[str, float]]:
        costs = {}
        for idx in range(4):
            c = self.evaluator.evaluate_trust_agnostic_cost(
                action_idx=idx,
                cognitive_load=cognitive_load,
                reliability=reliability,
                difficulty=difficulty,
                w_r=self.w_r,
                w_c=self.w_c,
            )
            costs[idx] = c

        best_idx = min(costs, key=costs.get)
        best_cost = costs[best_idx]
        prev_cost = costs[previous_action_idx]

        # Action smoothing / hysteresis threshold
        if abs(best_cost - prev_cost) < self.delta_switch:
            final_idx = previous_action_idx
        else:
            final_idx = best_idx

        return (
            final_idx,
            self.action_names[final_idx],
            costs[final_idx],
            {self.action_names[i]: costs[i] for i in range(4)},
        )


class TrustCalibratedPolicy(BaseExplanationPolicy):
    """
    Proposed Trust-Calibrated Adaptive Policy (pi_TC).
    Solves full asymmetric cost minimization: a_t^* = argmin_{a in A} J_t(a).
    """

    def __init__(
        self,
        cost_evaluator: Optional[ControllerCostEvaluator] = None,
        delta_switch: float = 0.15,
    ):
        self.evaluator = cost_evaluator or ControllerCostEvaluator()
        self.delta_switch = delta_switch
        self.action_names = ["E0", "E1", "E2", "E3"]

    def select_action(
        self,
        inferred_trust: float,
        cognitive_load: float,
        reliability: float,
        difficulty: float,
        previous_action_idx: int = 1,
    ) -> Tuple[int, str, float, Dict[str, float]]:
        costs = {}
        for idx in range(4):
            c = self.evaluator.evaluate_action_cost(
                action_idx=idx,
                inferred_trust=inferred_trust,
                cognitive_load=cognitive_load,
                reliability=reliability,
                difficulty=difficulty,
            )
            costs[idx] = c

        best_idx = min(costs, key=costs.get)
        best_cost = costs[best_idx]
        prev_cost = costs[previous_action_idx]

        # Action smoothing hysteresis to prevent visual fluttering
        if abs(best_cost - prev_cost) < self.delta_switch:
            final_idx = previous_action_idx
        else:
            final_idx = best_idx

        return (
            final_idx,
            self.action_names[final_idx],
            costs[final_idx],
            {self.action_names[i]: costs[i] for i in range(4)},
        )


class ControllerAblationPolicy(BaseExplanationPolicy):
    """
    Parametric ablation policy for isolating specific controller components:
    - mode="reliability_only": w1=0, w2=0, w5=0, w6=0
    - mode="trust_only": w2=0 (no cognitive load penalty)
    - mode="symmetric": w5=1.0, w6=1.0 (no overtrust/undertrust asymmetry)
    """

    def __init__(self, mode: str = "trust_only", delta_switch: float = 0.15):
        self.mode = mode
        self.delta_switch = delta_switch
        self.action_names = ["E0", "E1", "E2", "E3"]

        if mode == "reliability_only":
            evaluator = ControllerCostEvaluator(w1=0.0, w2=0.0, w3=0.30, w4=0.60, w5=0.0, w6=0.0)
        elif mode == "trust_only":
            evaluator = ControllerCostEvaluator(w1=1.00, w2=0.0, w3=0.30, w4=0.60, w5=1.20, w6=0.80)
        elif mode == "symmetric":
            evaluator = ControllerCostEvaluator(w1=1.00, w2=0.45, w3=0.30, w4=0.60, w5=1.00, w6=1.00)
        else:
            evaluator = ControllerCostEvaluator()
        self.evaluator = evaluator

    def select_action(
        self,
        inferred_trust: float,
        cognitive_load: float,
        reliability: float,
        difficulty: float,
        previous_action_idx: int = 1,
    ) -> Tuple[int, str, float, Dict[str, float]]:
        costs = {}
        for idx in range(4):
            c = self.evaluator.evaluate_action_cost(
                action_idx=idx,
                inferred_trust=inferred_trust,
                cognitive_load=cognitive_load,
                reliability=reliability,
                difficulty=difficulty,
            )
            costs[idx] = c

        best_idx = min(costs, key=costs.get)
        best_cost = costs[best_idx]
        prev_cost = costs[previous_action_idx]

        if abs(best_cost - prev_cost) < self.delta_switch:
            final_idx = previous_action_idx
        else:
            final_idx = best_idx

        return (
            final_idx,
            self.action_names[final_idx],
            costs[final_idx],
            {self.action_names[i]: costs[i] for i in range(4)},
        )
