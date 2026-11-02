"""
Closed-Loop Interaction Simulation Engine.
Implements Algorithm 1 ("Closed-Loop Trust-Calibrated Adaptive Explainability"):
Alert Ingestion -> State Prediction -> TCE Calculation -> Action Selection ->
Interface Rendering -> Analyst Decision & Telemetry -> EKF State Correction.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from ..state_estimation.ekf import AnalystStateEKF
from ..state_estimation.observation import BehavioralObservationModel
from ..controller.adaptive_engine import AdaptiveExplainabilityEngine
from ..explanations.generator import AdaptiveExplanationGenerator
from .alert_stream import AlertItem, AlertStreamGenerator
from .analyst_agent import SimulatedAnalystAgent


class ClosedLoopSimulation:
    """
    Executes the closed-loop human-AI interaction sequence across investigation trials.
    """

    def __init__(
        self,
        ekf: Optional[AnalystStateEKF] = None,
        obs_model: Optional[BehavioralObservationModel] = None,
        generator: Optional[AdaptiveExplanationGenerator] = None,
    ):
        self.ekf = ekf or AnalystStateEKF()
        self.obs_model = obs_model or BehavioralObservationModel()
        self.generator = generator or AdaptiveExplanationGenerator()

    def run_trial(
        self,
        alert: AlertItem,
        controller: AdaptiveExplainabilityEngine,
        analyst: SimulatedAnalystAgent,
        previous_action_idx: int = 1,
        previous_correct: Optional[bool] = None,
        previous_severity: float = 0.50,
    ) -> Dict[str, Any]:
        """
        Executes a single interaction step corresponding to Algorithm 1.
        """
        # 1. State Prediction Step
        # Predict prior analyst state: z_{t|t-1} = f(z_{t-1|t-1}, u_{t-1})
        prior_trust, prior_load = self.ekf.predict(
            action_idx=previous_action_idx,
            reliability=alert.calibrated_reliability,
            difficulty=alert.difficulty,
            severity=previous_severity,
            recommendation_correct=previous_correct,
        )

        # 2. Compute Instantaneous TCE_t and select action a_t^*
        adaptation_record = controller.step(
            trial_id=alert.trial_id,
            inferred_trust=prior_trust,
            cognitive_load=prior_load,
            reliability=alert.calibrated_reliability,
            difficulty=alert.difficulty,
        )
        action_name = adaptation_record["action_name"]
        action_idx = adaptation_record["action_idx"]

        # 3. Render Interface Explanation Payload
        explanation_payload = self.generator.generate(
            action_name=action_name,
            predicted_class=alert.nids_predicted_class,
            reliability=alert.calibrated_reliability,
            flow_indicators=alert.raw_indicators,
        )

        # 4. Analyst Interaction & Telemetry Logging
        analyst_response = analyst.decide(
            alert=alert,
            explanation_action=action_name,
            current_trust=prior_trust,
            current_load=prior_load,
        )

        # 5. Process Telemetry & Perform EKF Measurement Update
        telem = analyst_response["telemetry"]
        obs_vector = self.obs_model.process_raw_telemetry(
            gaze_dispersion=telem["gaze_dispersion"],
            gaze_entropy=telem["gaze_entropy"],
            head_angular_velocity=telem["head_movement"],
            decision_latency=telem["response_time"],
            verification_clicks=telem["verification_clicks"],
            decision_revisions=telem["revisions"],
        )
        posterior_trust, posterior_load = self.ekf.update(obs_vector)

        # 6. Assemble Trial Result
        return {
            "trial_id": alert.trial_id,
            "block_id": alert.block_id,
            "policy": controller.policy_name,
            "true_class": alert.true_class,
            "predicted_class": alert.nids_predicted_class,
            "reliability": alert.calibrated_reliability,
            "difficulty": alert.difficulty,
            "severity": alert.severity,
            "is_perturbation": alert.is_perturbation,
            "perturbation_type": alert.perturbation_type,
            "is_ai_correct": alert.is_correct_recommendation,
            "prior_trust": prior_trust,
            "prior_load": prior_load,
            "tce": adaptation_record["tce"],
            "action_idx": action_idx,
            "action_name": action_name,
            "action_cost": adaptation_record["action_cost"],
            "switched": adaptation_record["switched"],
            "analyst_verdict": analyst_response["analyst_verdict"],
            "accepted_ai": analyst_response["accepted_ai"],
            "is_correct_decision": analyst_response["is_correct_decision"],
            "response_time": analyst_response["response_time"],
            "inspection_time": analyst_response["inspection_time"],
            "verification_clicks": analyst_response["verification_clicks"],
            "revisions": analyst_response["revisions"],
            "self_reported_trust": analyst_response["self_reported_trust"],
            "posterior_trust": posterior_trust,
            "posterior_load": posterior_load,
            "explanation_payload": explanation_payload,
        }

    def run_participant_session(
        self,
        participant_id: int,
        stream_gen: Optional[AlertStreamGenerator] = None,
    ) -> Dict[str, Any]:
        """
        Runs full 60-trial session for one participant across the counterbalanced blocks.
        """
        stream_gen = stream_gen or AlertStreamGenerator(seed=participant_id * 100)
        alerts, condition_order = stream_gen.generate_participant_stream(participant_id)
        analyst = SimulatedAnalystAgent(participant_id=participant_id)

        all_trial_results: List[Dict[str, Any]] = []
        block_summaries: List[Dict[str, Any]] = []

        # 3 blocks of 20 trials
        for b_idx in range(3):
            block_id = b_idx + 1
            condition_name = condition_order[b_idx]
            controller = AdaptiveExplainabilityEngine(policy_name=condition_name)
            self.ekf.reset(initial_trust=0.50, initial_load=0.30)

            block_alerts = alerts[b_idx * 20 : (b_idx + 1) * 20]
            block_trials = []
            prev_action = 1
            prev_correct = None
            prev_sev = 0.50

            for alert in block_alerts:
                res = self.run_trial(
                    alert=alert,
                    controller=controller,
                    analyst=analyst,
                    previous_action_idx=prev_action,
                    previous_correct=prev_correct,
                    previous_severity=prev_sev,
                )
                res["participant_id"] = participant_id
                res["condition"] = condition_name
                block_trials.append(res)
                all_trial_results.append(res)
                prev_action = res["action_idx"]
                prev_correct = alert.is_correct_recommendation
                prev_sev = alert.severity

            # Compute Block NASA-TLX
            avg_load = float(np.mean([t["posterior_load"] for t in block_trials]))
            avg_inspect = float(np.mean([t["inspection_time"] for t in block_trials]))
            tlx = analyst.generate_block_tlx(condition_name, avg_load, avg_inspect)

            block_summaries.append({
                "participant_id": participant_id,
                "block_id": block_id,
                "condition": condition_name,
                "nasa_tlx": tlx,
                "mean_response_time": float(np.mean([t["response_time"] for t in block_trials])),
                "accuracy": float(np.mean([t["is_correct_decision"] for t in block_trials])),
            })

        return {
            "participant_id": participant_id,
            "condition_order": condition_order,
            "trial_results": all_trial_results,
            "block_summaries": block_summaries,
        }

    def run_full_cohort(
        self,
        num_participants: int = 36,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Runs the full 36-participant evaluation cohort generating 2,160 investigation trials.
        """
        all_trials = []
        all_blocks = []

        for p_id in range(1, num_participants + 1):
            session = self.run_participant_session(participant_id=p_id)
            all_trials.extend(session["trial_results"])
            all_blocks.extend(session["block_summaries"])
            if progress_callback:
                progress_callback(p_id, num_participants)

        return {
            "num_participants": num_participants,
            "total_trials": len(all_trials),
            "trials": all_trials,
            "blocks": all_blocks,
        }
