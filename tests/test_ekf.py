"""Tests for Extended Kalman Filter (EKF) and telemetry observation model."""

import unittest
import numpy as np
from src.state_estimation.ekf import AnalystStateEKF
from src.state_estimation.observation import BehavioralObservationModel


class TestEKF(unittest.TestCase):

    def test_ekf_initialization_and_bounds(self):
        ekf = AnalystStateEKF(initial_trust=0.50, initial_load=0.30)
        self.assertAlmostEqual(ekf.trust, 0.50, places=3)
        self.assertAlmostEqual(ekf.cognitive_load, 0.30, places=3)

    def test_ekf_prediction_negativity_bias(self):
        ekf_success = AnalystStateEKF(initial_trust=0.50, beta1=0.25, beta2=0.65)
        t_succ, _ = ekf_success.predict(recommendation_correct=True, reliability=0.80, severity=0.80)

        ekf_fail = AnalystStateEKF(initial_trust=0.50, beta1=0.25, beta2=0.65)
        t_fail, _ = ekf_fail.predict(recommendation_correct=False, reliability=0.80, severity=0.80)

        gain_success = t_succ - 0.50
        loss_fail = 0.50 - t_fail
        self.assertGreater(loss_fail, gain_success)

    def test_ekf_update_with_observation(self):
        ekf = AnalystStateEKF(initial_trust=0.50, initial_load=0.30)
        ekf.predict()

        obs_model = BehavioralObservationModel()
        obs = obs_model.process_raw_telemetry(
            gaze_dispersion=0.35,
            gaze_entropy=2.2,
            head_angular_velocity=0.20,
            decision_latency=45.0,
        )
        self.assertEqual(obs.shape, (4,))

        post_trust, post_load = ekf.update(obs)
        self.assertTrue(0.0 <= post_trust <= 1.0)
        self.assertTrue(0.0 <= post_load <= 1.0)
        self.assertTrue(np.allclose(ekf.P, ekf.P.T))
        self.assertTrue(np.all(np.linalg.eigvals(ekf.P) > 0.0))


if __name__ == "__main__":
    unittest.main()
