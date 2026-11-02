"""
Alert Stream and Stimulus Protocol Generator.
Synthesizes the 60-trial experimental sequences structured into 3 counterbalanced blocks of 20 trials,
matching the exact reliability distributions and perturbation injection protocols of the paper.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from ..explanations.knowledge_base import ThreatKnowledgeBase


class AlertItem:
    """Represents a single security event presented during an investigation trial."""

    def __init__(
        self,
        trial_id: int,
        block_id: int,
        true_class: str,
        nids_predicted_class: str,
        calibrated_reliability: float,
        difficulty: float,
        severity: float,
        is_perturbation: bool,
        perturbation_type: Optional[str] = None,
        flow_features: Optional[np.ndarray] = None,
        raw_indicators: Optional[List[str]] = None,
    ):
        self.trial_id = trial_id
        self.block_id = block_id
        self.true_class = true_class
        self.nids_predicted_class = nids_predicted_class
        self.calibrated_reliability = float(calibrated_reliability)
        self.difficulty = float(np.clip(difficulty, 0.05, 0.95))
        self.severity = float(np.clip(severity, 0.0, 1.0))
        self.is_perturbation = is_perturbation
        self.perturbation_type = perturbation_type
        self.flow_features = flow_features
        self.raw_indicators = raw_indicators or []

    @property
    def is_correct_recommendation(self) -> bool:
        return self.true_class == self.nids_predicted_class

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "block_id": self.block_id,
            "true_class": self.true_class,
            "nids_predicted_class": self.nids_predicted_class,
            "calibrated_reliability": self.calibrated_reliability,
            "difficulty": self.difficulty,
            "severity": self.severity,
            "is_correct_recommendation": self.is_correct_recommendation,
            "is_perturbation": self.is_perturbation,
            "perturbation_type": self.perturbation_type,
            "raw_indicators": self.raw_indicators,
        }


class AlertStreamGenerator:
    """
    Generates experimental blocks (S1, S2, S3) matching:
    - 20 trials per block (10 Benign, 10 Malicious)
    - 12 High reliability [0.80, 0.95]
    - 5 Medium reliability [0.55, 0.79]
    - 3 Low reliability [0.30, 0.54]
    - 2 Injected perturbation errors per block (1 high-confidence, 1 low-confidence)
    """

    def __init__(self, kb: Optional[ThreatKnowledgeBase] = None, seed: int = 42):
        self.kb = kb or ThreatKnowledgeBase()
        self.rng = np.random.default_rng(seed)
        self.malicious_classes = [c for c in self.kb.classes.keys() if c != "Benign"]

    def compute_difficulty(
        self,
        reliability: float,
        num_indicators: int,
        is_cross_subnet: bool,
    ) -> float:
        """
        Calculates task difficulty d_t in [0, 1] as defined in Section 4.2 / Action P-06:
        d_t = omega1 * H(p_t) + omega2 * (num_indicators / K_max) + omega3 * I(cross_subnet)
        """
        # Predictive entropy proxy
        p = np.clip(reliability, 1e-4, 1.0 - 1e-4)
        entropy = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
        norm_entropy = float(entropy)  # Max binary entropy is 1.0

        indicator_ratio = min(1.0, num_indicators / 5.0)
        subnet_term = 1.0 if is_cross_subnet else 0.0

        d_t = 0.50 * norm_entropy + 0.30 * indicator_ratio + 0.20 * subnet_term
        return float(np.clip(d_t, 0.10, 0.95))

    def generate_block(
        self,
        block_id: int,
        start_trial_id: int = 1,
        seed: Optional[int] = None,
    ) -> List[AlertItem]:
        """Generates a structured 20-trial experimental block."""
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        alerts: List[AlertItem] = []

        # 12 High [0.80, 0.95], 5 Medium [0.55, 0.79], 3 Low [0.30, 0.54]
        high_rel = rng.uniform(0.80, 0.95, size=12)
        med_rel = rng.uniform(0.55, 0.79, size=5)
        low_rel = rng.uniform(0.30, 0.54, size=3)
        all_reliabilities = np.concatenate([high_rel, med_rel, low_rel])
        rng.shuffle(all_reliabilities)

        # 10 Benign, 10 Malicious
        is_malicious = [False] * 10 + [True] * 10
        rng.shuffle(is_malicious)

        # Perturbation indices: 2 errors per block (1 high-confidence, 1 low/medium-confidence)
        # Select positions not on trial 0 or trial 1 to allow initial stabilization
        possible_high_idx = [i for i, r in enumerate(all_reliabilities) if r >= 0.80 and i >= 3]
        possible_low_idx = [i for i, r in enumerate(all_reliabilities) if r < 0.60 and i >= 4]

        pert_high_idx = rng.choice(possible_high_idx) if possible_high_idx else 6
        remaining_low = [i for i in possible_low_idx if i != pert_high_idx]
        pert_low_idx = rng.choice(remaining_low) if remaining_low else 14

        for i in range(20):
            trial_num = start_trial_id + i
            rel = float(all_reliabilities[i])
            mal = is_malicious[i]

            is_pert = i in [pert_high_idx, pert_low_idx]
            pert_type = None

            if mal:
                true_c = rng.choice(self.malicious_classes)
            else:
                true_c = "Benign"

            if is_pert:
                if i == pert_high_idx:
                    pert_type = "high_confidence_false_positive_or_evasion"
                    # Model mistakenly recommends wrong category with high confidence
                    pred_c = rng.choice([c for c in self.kb.classes.keys() if c != true_c])
                    rel = max(rel, 0.85)  # Overtrust stressor
                else:
                    pert_type = "low_confidence_misclassification"
                    pred_c = rng.choice([c for c in self.kb.classes.keys() if c != true_c])
                    rel = min(rel, 0.48)  # Ambiguity stressor
            else:
                pred_c = true_c  # Correct recommendation

            info = self.kb.get_class_info(true_c)
            indicators = info.get("indicators", ["Normal traffic flow"])
            is_cross = (true_c in ["Infiltration", "Botnet", "Worms", "Ransomware"])
            diff = self.compute_difficulty(rel, len(indicators), is_cross)
            sev = self.kb.get_severity(true_c)

            item = AlertItem(
                trial_id=trial_num,
                block_id=block_id,
                true_class=true_c,
                nids_predicted_class=pred_c,
                calibrated_reliability=rel,
                difficulty=diff,
                severity=sev,
                is_perturbation=is_pert,
                perturbation_type=pert_type,
                raw_indicators=indicators,
            )
            alerts.append(item)

        return alerts

    def generate_participant_stream(
        self,
        participant_id: int,
    ) -> Tuple[List[AlertItem], List[str]]:
        """
        Generates full 60-trial stream partitioned into 3 blocks (S1, S2, S3).
        Applies 3x3 Latin Square counterbalancing for interface conditions across 6 administration sequences.
        """
        sequences = [
            ["Static", "Trust-Agnostic", "Trust-Calibrated"],
            ["Static", "Trust-Calibrated", "Trust-Agnostic"],
            ["Trust-Agnostic", "Static", "Trust-Calibrated"],
            ["Trust-Agnostic", "Trust-Calibrated", "Static"],
            ["Trust-Calibrated", "Static", "Trust-Agnostic"],
            ["Trust-Calibrated", "Trust-Agnostic", "Static"],
        ]
        seq_idx = (participant_id - 1) % len(sequences)
        condition_order = sequences[seq_idx]

        stream: List[AlertItem] = []
        seed_base = participant_id * 100

        for b_id in range(1, 4):
            block_alerts = self.generate_block(
                block_id=b_id,
                start_trial_id=(b_id - 1) * 20 + 1,
                seed=seed_base + b_id,
            )
            stream.extend(block_alerts)

        return stream, condition_order
