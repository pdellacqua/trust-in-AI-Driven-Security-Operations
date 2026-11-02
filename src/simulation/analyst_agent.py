"""
Simulated Human Analyst Agent.
Stochastically simulates analyst decision behavior, appropriate reliance,
investigation latency, and passive ocular/telemetry indicators based on cognitive principles.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from ..state_estimation.ekf import sigmoid, logit
from .alert_stream import AlertItem


class SimulatedAnalystAgent:
    """
    Simulates a human security analyst with latent cognitive states:
    - Inherent baseline trust and expertise
    - Dynamic susceptibility to automation bias and inappropriate rejection
    - Telemetry generator producing realistic visual and interaction signals
    """

    def __init__(
        self,
        participant_id: int,
        expertise_level: float = 0.85,
        base_reaction_speed: float = 28.0,
        seed: Optional[int] = None,
    ):
        self.participant_id = participant_id
        self.expertise = expertise_level
        self.base_speed = base_reaction_speed
        self.rng = np.random.default_rng(seed or (participant_id * 31 + 7))

        # Personal propensity offsets
        self.trust_bias = float(self.rng.normal(0.0, 0.15))
        self.speed_bias = float(self.rng.normal(0.0, 3.5))

    def decide(
        self,
        alert: AlertItem,
        explanation_action: str,
        current_trust: float,
        current_load: float,
    ) -> Dict[str, Any]:
        """
        Simulates analyst decision on alert given explanation level and cognitive state.
        Returns decision verdict, acceptance flag, response time, and behavioral telemetry.
        """
        action_intensity_map = {"E0": 0.10, "E1": 0.40, "E2": 0.70, "E3": 1.00}
        kappa_a = action_intensity_map.get(explanation_action, 0.40)

        is_ai_correct = alert.is_correct_recommendation
        r_t = alert.calibrated_reliability
        d_t = alert.difficulty

        # Effective trust governing behavior
        eff_trust = np.clip(current_trust + self.trust_bias, 0.05, 0.95)

        # 1. Behavioral Reliance Determination
        if not is_ai_correct:
            # Automation Bias scenario: AI recommendation is WRONG
            # High trust and high model confidence push toward inappropriate acceptance.
            # Explanations E2/E3 expose conflicting evidence and enable verification.
            p_accept_logits = (
                -0.28
                + 2.5 * (eff_trust - 0.50)
                + 1.8 * (r_t - 0.50)
                + 0.6 * (current_load - 0.30)
                - 2.8 * (kappa_a - 0.40)
            )
            p_accept = sigmoid(p_accept_logits)
            accepted = bool(self.rng.random() < p_accept)

            if accepted:
                analyst_verdict = alert.nids_predicted_class
                is_correct_decision = False
            else:
                # Analyst rejected error; probability of correctly identifying true class
                p_recover = float(np.clip(0.70 + 0.20 * (kappa_a - 0.40) + 0.10 * (self.expertise - 0.80), 0.50, 0.95))
                if self.rng.random() < p_recover:
                    analyst_verdict = alert.true_class
                    is_correct_decision = True
                else:
                    analyst_verdict = "Unknown / Suspicious"
                    is_correct_decision = False
        else:
            # Appropriate Reliance scenario: AI recommendation is CORRECT
            # Low trust, high task difficulty, or cognitive strain can induce inappropriate rejection (IRR).
            # Explanations E1/E2 provide supporting evidence, reducing unjustified skepticism.
            p_reject_logits = (
                -1.46
                - 2.8 * (eff_trust - 0.50)
                - 2.0 * (r_t - 0.50)
                + 1.2 * (d_t - 0.40)
                + 0.5 * (current_load - 0.30)
                - 2.2 * (kappa_a - 0.40)
            )
            p_reject = sigmoid(p_reject_logits)
            rejected = bool(self.rng.random() < p_reject)

            if rejected:
                accepted = False
                analyst_verdict = "Benign" if alert.true_class != "Benign" else "Other Attack"
                is_correct_decision = False
            else:
                accepted = True
                analyst_verdict = alert.nids_predicted_class
                is_correct_decision = True

        # 2. Response Time & Interaction Latency
        # Routine high-confidence alerts with E0/E1 are fast; complex E3 alerts take longer.
        base_time = self.base_speed + self.speed_bias
        explanation_inspection_time = 4.0 * kappa_a + self.rng.uniform(1.0, 3.5)
        difficulty_delay = d_t * 6.5
        verification_clicks = 0
        revisions = 0

        if not is_ai_correct and not accepted:
            # Extra verification time spent rejecting incorrect AI
            difficulty_delay += self.rng.uniform(5.0, 9.0)
            verification_clicks = int(self.rng.integers(1, 4))
            revisions = 1
        elif is_ai_correct and not accepted:
            difficulty_delay += self.rng.uniform(3.0, 7.0)
            verification_clicks = int(self.rng.integers(1, 3))
            revisions = 1

        total_response_time = max(8.0, base_time + explanation_inspection_time + difficulty_delay)

        # 3. Behavioral Telemetry Signals (OpenFace & interaction proxy)
        # Gaze dispersion and entropy correlate with cognitive load and search activity
        gaze_dispersion = float(0.25 + 0.12 * current_load + self.rng.normal(0, 0.02))
        gaze_entropy = float(1.80 + 0.45 * current_load - 0.15 * eff_trust + self.rng.normal(0, 0.05))
        head_movement = float(0.15 + 0.08 * current_load + self.rng.normal(0, 0.02))

        # 4. Self-Reported Trust Probe (administered periodically on 7-point Likert scale normalized to [0, 1])
        self_reported_trust = float(np.clip(eff_trust + self.rng.normal(0, 0.06), 0.0, 1.0))

        return {
            "analyst_verdict": analyst_verdict,
            "accepted_ai": accepted,
            "is_correct_decision": is_correct_decision,
            "response_time": float(total_response_time),
            "inspection_time": float(explanation_inspection_time),
            "verification_clicks": verification_clicks,
            "revisions": revisions,
            "self_reported_trust": self_reported_trust,
            "telemetry": {
                "gaze_dispersion": gaze_dispersion,
                "gaze_entropy": gaze_entropy,
                "head_movement": head_movement,
                "response_time": total_response_time,
                "verification_clicks": verification_clicks,
                "revisions": revisions,
            },
        }

    def generate_block_tlx(
        self,
        condition_name: str,
        avg_block_load: float,
        avg_inspection_time: float,
    ) -> float:
        """
        Simulates NASA-TLX subjective cognitive workload score [0, 100] at end of a 20-trial block.
        """
        if condition_name == "Static":
            base_tlx = 58.4
        elif condition_name in ["Trust-Agnostic", "TA"]:
            base_tlx = 54.1
        else:  # Trust-Calibrated
            base_tlx = 48.7

        load_offset = (avg_block_load - 0.35) * 15.0
        tlx_score = base_tlx + load_offset + float(self.rng.normal(0.0, 3.5))
        return float(np.clip(tlx_score, 20.0, 95.0))
