#!/usr/bin/env python3
"""
Controller Ablation Study Runner.
Evaluates:
- Full Trust-Calibrated (pi_TC)
- Trust-Only (Ablation of Cognitive Load Modeling, w2=0)
- Reliability-Only (Ablation of Latent State Feedback, w1=w2=w5=w6=0)
- Symmetric Penalty (w5 = w6 = 1.00)
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.simulation.closed_loop import ClosedLoopSimulation
from src.simulation.alert_stream import AlertStreamGenerator
from src.simulation.analyst_agent import SimulatedAnalystAgent
from src.controller.adaptive_engine import AdaptiveExplainabilityEngine
from src.controller.policies import ControllerAblationPolicy, TrustCalibratedPolicy
from src.evaluation.metrics import compute_condition_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Run Controller Ablations")
    parser.add_argument("--participants", type=int, default=12, help="Number of participants for ablation")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    print(f"\n=======================================================")
    print(f" Controller Component Ablation Analysis")
    print(f" Cohort Size: {args.participants} participants")
    print(f"=======================================================\n")

    ablation_modes = [
        ("Full Proposed TC", TrustCalibratedPolicy()),
        ("Trust-Only (No Workload)", ControllerAblationPolicy(mode="trust_only")),
        ("Reliability-Only", ControllerAblationPolicy(mode="reliability_only")),
        ("Symmetric Penalty (No Asymm.)", ControllerAblationPolicy(mode="symmetric")),
    ]

    results = {}
    gen = AlertStreamGenerator(seed=args.seed)

    for name, policy in ablation_modes:
        print(f"[*] Evaluating ablation configuration: {name}...")
        sim = ClosedLoopSimulation()
        trials_list = []

        for p_id in range(1, args.participants + 1):
            alerts, _ = gen.generate_participant_stream(p_id)
            analyst = SimulatedAnalystAgent(participant_id=p_id, seed=args.seed + p_id)
            controller = AdaptiveExplainabilityEngine(policy_name=name, policy=policy)
            sim.ekf.reset(initial_trust=0.50, initial_load=0.30)

            prev_act = 1
            prev_corr = None
            prev_sev = 0.50

            for alert in alerts[:40]:  # 2 blocks evaluation
                trial_res = sim.run_trial(
                    alert=alert,
                    controller=controller,
                    analyst=analyst,
                    previous_action_idx=prev_act,
                    previous_correct=prev_corr,
                    previous_severity=prev_sev,
                )
                trial_res["condition"] = name
                trials_list.append(trial_res)
                prev_act = trial_res["action_idx"]
                prev_corr = alert.is_correct_recommendation
                prev_sev = alert.severity

        metrics = compute_condition_metrics(trials_list)
        results[name] = metrics

    print(f"\n==========================================================================")
    print(f" TABLE: Controller Ablation Results")
    print(f"==========================================================================")
    print(f"{'Configuration':<30} | {'MACE':<14} | {'Accuracy (%)':<14} | {'ABR (%)':<10}")
    print(f"-" * 74)
    for name, m in results.items():
        print(f"{name:<30} | {m['mace']:.3f} +/- {m['mace_std']:.3f} | {m['accuracy']*100:.1f}%        | {m['abr']*100:.1f}%")
    print(f"==========================================================================\n")


if __name__ == "__main__":
    main()
