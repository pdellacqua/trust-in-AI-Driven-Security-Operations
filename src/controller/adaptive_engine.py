"""
Adaptive Engine Orchestrator.
Manages the online explanation adaptation cycle, telemetry recording, and state hysteresis.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from .policies import (
    BaseExplanationPolicy,
    StaticExplanationPolicy,
    TrustAgnosticAdaptivePolicy,
    TrustCalibratedPolicy,
)


class AdaptiveExplainabilityEngine:
    """
    Main controller engine managing explanation selection and interaction history.
    """

    def __init__(
        self,
        policy_name: str = "Trust-Calibrated",
        policy: Optional[BaseExplanationPolicy] = None,
    ):
        self.policy_name = policy_name
        if policy is not None:
            self.policy = policy
        elif policy_name in ["Static", "S"]:
            self.policy = StaticExplanationPolicy(fixed_level_idx=1)
        elif policy_name in ["Trust-Agnostic", "TA"]:
            self.policy = TrustAgnosticAdaptivePolicy()
        else:
            self.policy = TrustCalibratedPolicy()

        self.current_action_idx = 1
        self.current_action_name = "E1"
        self.history: List[Dict[str, any]] = []

    def reset(self, initial_action_idx: int = 1):
        """Resets engine state."""
        self.current_action_idx = initial_action_idx
        self.current_action_name = ["E0", "E1", "E2", "E3"][initial_action_idx]
        self.history = []

    def step(
        self,
        trial_id: int,
        inferred_trust: float,
        cognitive_load: float,
        reliability: float,
        difficulty: float,
    ) -> Dict[str, any]:
        """
        Executes one adaptation step:
        1. Computes instantaneous TCE_t = T_hat_t - r_t
        2. Evaluates explanation costs
        3. Applies hysteresis and returns selected level
        """
        tce = inferred_trust - reliability
        action_idx, action_name, cost, all_costs = self.policy.select_action(
            inferred_trust=inferred_trust,
            cognitive_load=cognitive_load,
            reliability=reliability,
            difficulty=difficulty,
            previous_action_idx=self.current_action_idx,
        )

        switched = action_idx != self.current_action_idx
        self.current_action_idx = action_idx
        self.current_action_name = action_name

        record = {
            "trial_id": trial_id,
            "policy": self.policy_name,
            "inferred_trust": inferred_trust,
            "cognitive_load": cognitive_load,
            "reliability": reliability,
            "difficulty": difficulty,
            "tce": tce,
            "action_idx": action_idx,
            "action_name": action_name,
            "action_cost": cost,
            "all_costs": all_costs,
            "switched": switched,
        }
        self.history.append(record)
        return record
