"""
Telemetry Processing and Observation Feature Extraction.
Maps webcam-derived ocular/head cues and UI interaction logs into normalized observations o_t.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np


class BehavioralObservationModel:
    """
    Extracts, standardizes, and packages behavioral and interaction features:
    o_t = [gaze_dispersion, gaze_entropy, head_movement, interaction_latency]^T
    """

    def __init__(self, baseline_stats: Optional[Dict[str, Tuple[float, float]]] = None):
        # Baseline mean and standard deviation per feature for subject-specific normalization
        self.baseline_stats = baseline_stats or {
            "gaze_dispersion": (0.25, 0.08),
            "gaze_entropy": (1.80, 0.35),
            "head_movement": (0.15, 0.06),
            "interaction_latency": (28.0, 8.5),
        }

    def process_raw_telemetry(
        self,
        gaze_dispersion: float,
        gaze_entropy: float,
        head_angular_velocity: float,
        decision_latency: float,
        verification_clicks: int = 0,
        decision_revisions: int = 0,
    ) -> np.ndarray:
        """
        Converts raw visual and interaction indicators into standardized z-score observation vector.
        """
        # Composite latency metric incorporating verification delays and revisions
        composite_latency = decision_latency + (verification_clicks * 2.5) + (decision_revisions * 4.0)

        # Standardize features against subject baseline
        z1 = (gaze_dispersion - self.baseline_stats["gaze_dispersion"][0]) / self.baseline_stats["gaze_dispersion"][1]
        z2 = (gaze_entropy - self.baseline_stats["gaze_entropy"][0]) / self.baseline_stats["gaze_entropy"][1]
        z3 = (head_angular_velocity - self.baseline_stats["head_movement"][0]) / self.baseline_stats["head_movement"][1]
        z4 = (composite_latency - self.baseline_stats["interaction_latency"][0]) / self.baseline_stats["interaction_latency"][1]

        # Clip extreme outliers
        obs = np.array([z1, z2, z3, z4], dtype=np.float64)
        return np.clip(obs, -3.5, 3.5)

    def synthesize_noisy_observation(
        self,
        true_trust: float,
        true_load: float,
        W_o: np.ndarray,
        R: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Generates synthetic observation vector for simulated experiments:
        o_t = W_o * [logit(T_t), C_t]^T + v_t, v_t ~ N(0, R)
        """
        if rng is None:
            rng = np.random.default_rng()

        from .ekf import logit
        theta = logit(true_trust)
        z = np.array([theta, true_load], dtype=np.float64)
        mean_obs = W_o @ z
        noise = rng.multivariate_normal(mean=np.zeros(len(R)), cov=R)
        return mean_obs + noise
