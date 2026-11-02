"""
Baseline State Estimators for Comparative Benchmarking and Ablation Studies.
Implements:
- Constant Prior Estimator (no dynamic tracking)
- Moving Average Estimator (fixed sliding window)
- Linear Kalman Filter (without logistic transformation)
- Single-State Trust-Only EKF (ablation without cognitive load)
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from .ekf import sigmoid, logit


class ConstantPriorEstimator:
    """Baseline tracking model with constant neutral priors."""

    def __init__(self, prior_trust: float = 0.50, prior_load: float = 0.30):
        self._trust = prior_trust
        self._load = prior_load

    @property
    def trust(self) -> float:
        return self._trust

    @property
    def cognitive_load(self) -> float:
        return self._load

    def predict(self, *args, **kwargs) -> Tuple[float, float]:
        return self._trust, self._load

    def update(self, *args, **kwargs) -> Tuple[float, float]:
        return self._trust, self._load


class MovingAverageEstimator:
    """Sliding-window moving average estimator over recommendation outcomes."""

    def __init__(self, window_size: int = 3, initial_trust: float = 0.50, initial_load: float = 0.30):
        self.window_size = window_size
        self.history: List[float] = [initial_trust] * window_size
        self.load_history: List[float] = [initial_load] * window_size

    @property
    def trust(self) -> float:
        return float(np.mean(self.history))

    @property
    def cognitive_load(self) -> float:
        return float(np.mean(self.load_history))

    def predict(self, *args, **kwargs) -> Tuple[float, float]:
        return self.trust, self.cognitive_load

    def update(self, outcome: float, latency: float = 28.0) -> Tuple[float, float]:
        self.history.append(float(outcome))
        if len(self.history) > self.window_size:
            self.history.pop(0)

        norm_load = float(np.clip(latency / 50.0, 0.0, 1.0))
        self.load_history.append(norm_load)
        if len(self.load_history) > self.window_size:
            self.load_history.pop(0)

        return self.trust, self.cognitive_load


class SingleStateTrustOnlyEKF:
    """
    1D EKF estimating only latent trust theta_t without modeling cognitive load C_t.
    Used for the cognitive load ablation analysis in Section 5.5.
    """

    def __init__(
        self,
        alpha: float = 0.92,
        beta1: float = 0.25,
        beta2: float = 0.65,
        beta3: float = 0.20,
        q_theta: float = 0.04,
        initial_trust: float = 0.50,
    ):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.q = q_theta
        self.theta = logit(initial_trust)
        self.P = 0.20
        self.theta_prior = self.theta
        self.P_prior = self.P
        # 1D observation mapping from latency
        self.w_o = -0.70
        self.r = 0.15

    @property
    def trust(self) -> float:
        return sigmoid(self.theta)

    @property
    def cognitive_load(self) -> float:
        return 0.30  # Fixed nominal workload

    def predict(self, recommendation_correct: Optional[bool] = None, reliability: float = 0.85, severity: float = 0.5):
        k_t = 1.0 if recommendation_correct else 0.0
        self.theta_prior = self.alpha * self.theta + self.beta1 * k_t - self.beta2 * (1.0 - k_t) * severity + self.beta3 * (reliability - 0.5)
        self.P_prior = (self.alpha ** 2) * self.P + self.q
        return sigmoid(self.theta_prior), 0.30

    def update(self, observation: np.ndarray) -> Tuple[float, float]:
        # Uses 4th dimension (latency proxy)
        obs_scalar = float(observation[3] if len(observation) > 3 else observation[0])
        innov = obs_scalar - (self.w_o * self.theta_prior)
        S = (self.w_o ** 2) * self.P_prior + self.r
        K = self.P_prior * self.w_o / S
        self.theta = self.theta_prior + K * innov
        self.P = (1.0 - K * self.w_o) * self.P_prior
        return self.trust, 0.30
