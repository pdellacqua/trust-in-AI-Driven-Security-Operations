"""
Extended Kalman Filter (EKF) for Joint Estimation of Latent Trust and Cognitive Load.
Tracks bivariate cognitive state z_t = [theta_t, C_t]^T from non-intrusive behavioral telemetry.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np


def sigmoid(x: float) -> float:
    """Logistic mapping sigma(theta) = 1 / (1 + exp(-theta))."""
    if x >= 0:
        z = np.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = np.exp(x)
        return z / (1.0 + z)


def logit(p: float) -> float:
    """Inverse sigmoid mapping."""
    p = np.clip(p, 1e-5, 1.0 - 1e-5)
    return float(np.log(p / (1.0 - p)))


class AnalystStateEKF:
    """
    Probabilistic state-space filter estimating latent trust (theta_t) and cognitive load (C_t).
    As formulated in Equations (15)-(32) and Table 1 of the paper.
    """

    def __init__(
        self,
        alpha: float = 0.92,
        rho: float = 0.88,
        beta1: float = 0.25,
        beta2: float = 0.65,
        beta3: float = 0.20,
        gamma1: float = 0.15,
        gamma2: float = 0.12,
        gamma3: float = 0.10,
        q_theta: float = 0.04,
        q_c: float = 0.05,
        r_diag: Optional[List[float]] = None,
        initial_trust: float = 0.50,
        initial_load: float = 0.30,
    ):
        # Persistence coefficients
        self.alpha = alpha
        self.rho = rho

        # Trust dynamics parameters (enforcing beta2 > beta1 for negativity bias)
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3

        # Cognitive load dynamics parameters
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3

        # State transition matrix F
        self.F = np.array([[self.alpha, 0.0], [0.0, self.rho]], dtype=np.float64)

        # Process noise covariance Q
        self.Q = np.diag([q_theta, q_c]).astype(np.float64)

        # Observation noise covariance R (4-dimensional telemetry)
        if r_diag is None:
            r_diag = [0.08, 0.12, 0.10, 0.15]
        self.R = np.diag(r_diag).astype(np.float64)

        # Observation mapping matrix W_o: maps [theta, C] -> 4D observation vector
        # [gaze_dispersion, gaze_entropy, head_movement, interaction_latency]
        self.W_o = np.array(
            [
                [-0.15, 0.55],   # gaze dispersion increases with load, decreases with high trust
                [-0.20, 0.60],   # gaze entropy increases with mental strain/search
                [-0.10, 0.40],   # head movement / instability
                [-0.70, 0.50],   # latency decreases with high trust, increases with load
            ],
            dtype=np.float64,
        )

        # Observation offset c
        self.c = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # Initial state: theta_0 = logit(initial_trust), C_0 = initial_load
        theta_0 = logit(initial_trust)
        self.z = np.array([theta_0, initial_load], dtype=np.float64)
        self.P = np.diag([0.20, 0.20]).astype(np.float64)

        # Prior predicted state
        self.z_prior = self.z.copy()
        self.P_prior = self.P.copy()

        # Action intensity impact weights
        self.action_impact_weights = np.array([0.0, 0.05, 0.10, 0.15], dtype=np.float64)
        self.action_workload_costs = np.array([0.0, 0.10, 0.25, 0.45], dtype=np.float64)

    def reset(self, initial_trust: float = 0.50, initial_load: float = 0.30):
        """Resets filter to neutral initial state."""
        theta_0 = logit(initial_trust)
        self.z = np.array([theta_0, initial_load], dtype=np.float64)
        self.P = np.diag([0.20, 0.20]).astype(np.float64)
        self.z_prior = self.z.copy()
        self.P_prior = self.P.copy()

    @property
    def trust(self) -> float:
        """Returns current filtered posterior trust T_t in [0, 1]."""
        return sigmoid(self.z[0])

    @property
    def cognitive_load(self) -> float:
        """Returns current filtered posterior cognitive load C_t in [0, 1]."""
        return float(np.clip(self.z[1], 0.0, 1.0))

    @property
    def prior_trust(self) -> float:
        """Returns prior predicted trust T_{t|t-1} in [0, 1]."""
        return sigmoid(self.z_prior[0])

    @property
    def prior_cognitive_load(self) -> float:
        """Returns prior predicted cognitive load C_{t|t-1} in [0, 1]."""
        return float(np.clip(self.z_prior[1], 0.0, 1.0))

    def predict(
        self,
        action_idx: int = 1,
        reliability: float = 0.85,
        difficulty: float = 0.30,
        recommendation_correct: Optional[bool] = None,
        severity: float = 0.50,
    ) -> Tuple[float, float]:
        """
        EKF Prediction Step:
        Projects latent state forward:
        theta_{t+1} = alpha * theta_t + beta1 * k_t - beta2 * (1 - k_t) * s_t + beta3 * r_t + eta * g(a_t)
        C_{t+1} = rho * C_t + gamma1 * d_t + gamma2 * E(a_t) + gamma3 * (1 - k_t)
        """
        # Determine recommendation success indicator k_t
        if recommendation_correct is not None:
            k_t = 1.0 if recommendation_correct else 0.0
        else:
            # SOC runtime mode: expected success given calibrated reliability r_t
            k_t = float(reliability)

        # Action impact with diminishing marginal returns
        action_intensity = self.action_impact_weights[action_idx]
        g_action = float(np.tanh(action_intensity))
        workload_action = self.action_workload_costs[action_idx]

        # Trust transition
        theta_next = (
            self.alpha * self.z[0]
            + self.beta1 * k_t
            - self.beta2 * (1.0 - k_t) * severity
            + self.beta3 * (reliability - 0.5)
            + g_action
        )

        # Cognitive load transition
        c_next = (
            self.rho * self.z[1]
            + self.gamma1 * difficulty
            + self.gamma2 * workload_action
            + self.gamma3 * (1.0 - k_t)
        )
        c_next = np.clip(c_next, 0.0, 1.0)

        self.z_prior = np.array([theta_next, c_next], dtype=np.float64)
        self.P_prior = self.F @ self.P @ self.F.T + self.Q

        return self.prior_trust, self.prior_cognitive_load

    def update(self, observation: np.ndarray) -> Tuple[float, float]:
        """
        EKF Correction Step:
        Incorporates behavioral and interaction telemetry o_t.
        """
        obs = np.asarray(observation, dtype=np.float64)
        if obs.ndim == 1:
            obs = obs.reshape(-1, 1)

        # Innovation: y = o_t - (W_o * z_{t|t-1} + c)
        pred_obs = (self.W_o @ self.z_prior.reshape(-1, 1)) + self.c.reshape(-1, 1)
        innovation = obs - pred_obs

        # Innovation covariance: S = W_o * P_{prior} * W_o^T + R
        S = self.W_o @ self.P_prior @ self.W_o.T + self.R

        # Kalman gain: K = P_{prior} * W_o^T * S^{-1}
        K = self.P_prior @ self.W_o.T @ np.linalg.inv(S)

        # State update: z_{t|t} = z_{t|t-1} + K * innovation
        z_new = self.z_prior.reshape(-1, 1) + K @ innovation
        self.z = z_new.flatten()
        self.z[1] = np.clip(self.z[1], 0.0, 1.0)  # Bound cognitive workload

        # Covariance update: P_{t|t} = (I - K * W_o) * P_{t|t-1}
        I = np.eye(2, dtype=np.float64)
        self.P = (I - K @ self.W_o) @ self.P_prior

        return self.trust, self.cognitive_load
