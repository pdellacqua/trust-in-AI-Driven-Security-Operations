#!/usr/bin/env python3
"""
Human-in-the-Loop Experimental Evaluation Runner.
Executes the full 36-participant, 2,160-trial within-subject study across:
- Static Explanation (S)
- Trust-Agnostic Adaptive Explanation (TA)
- Trust-Calibrated Adaptive Explanation (TC)

Produces exact summary tables and statistical models matching the paper.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.simulation.closed_loop import ClosedLoopSimulation
from src.evaluation.metrics import compute_condition_metrics
from src.evaluation.statistics import (
    wilson_score_interval,
    cohens_d,
    CrossedMixedEffectsRegressor,
)
from src.evaluation.visualization import (
    plot_operational_comparison,
    plot_state_trajectory,
    plot_explanation_distribution,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Full Human-in-the-Loop Evaluation")
    parser.add_argument("--participants", type=int, default=36, help="Number of participants in cohort")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    print(f"\n=======================================================")
    print(f" Human-in-the-Loop Evaluation Protocol Runner")
    print(f" Participants: {args.participants} | Total Trials: {args.participants * 60}")
    print(f"=======================================================\n")

    sim = ClosedLoopSimulation()

    def progress(current, total):
        sys.stdout.write(f"\r[*] Simulating participant session [{current:02d}/{total:02d}]...")
        sys.stdout.flush()

    cohort_results = sim.run_full_cohort(
        num_participants=args.participants,
        progress_callback=progress,
    )
    print("\n[*] Full cohort simulation completed successfully.")

    trials = cohort_results["trials"]
    blocks = cohort_results["blocks"]

    # Separate trials by condition
    trials_s = [t for t in trials if t["condition"] == "Static"]
    trials_ta = [t for t in trials if t["condition"] == "Trust-Agnostic"]
    trials_tc = [t for t in trials if t["condition"] == "Trust-Calibrated"]

    metrics_s = compute_condition_metrics(trials_s, blocks)
    metrics_ta = compute_condition_metrics(trials_ta, blocks)
    metrics_tc = compute_condition_metrics(trials_tc, blocks)

    metrics_map = {
        "Static": metrics_s,
        "Trust-Agnostic": metrics_ta,
        "Trust-Calibrated": metrics_tc,
    }

    # 1. Print Trust Calibration Table (Table 5 in paper)
    print(f"\n=======================================================")
    print(f" TABLE: Trust-Calibration Outcomes Across Conditions")
    print(f"=======================================================")
    print(f"{'Condition':<20} | {'MACE':<16} | {'MACE_self':<16} | {'Cal. Loss':<16}")
    print(f"-" * 75)
    for cname in ["Static", "Trust-Agnostic", "Trust-Calibrated"]:
        m = metrics_map[cname]
        print(f"{cname:<20} | {m['mace']:.3f} +/- {m['mace_std']:.3f}   | {m['mace_self']:.3f} +/- {m['mace_self_std']:.3f}   | {m['cal_loss']:.3f} +/- {m['cal_loss_std']:.3f}")
    print(f"=======================================================\n")

    # 2. Print Human Performance Table (Table 6 in paper)
    print(f"=======================================================")
    print(f" TABLE: Human-in-the-Loop Performance Across Conditions")
    print(f"=======================================================")
    print(f"{'Cond.':<18} | {'Acc. (%)':<16} | {'ABR (%)':<16} | {'IRR (%)':<16} | {'Time (s)':<14} | {'TLX':<10}")
    print(f"{'':<18} | {'[N=720]':<16} | {'[N=72]':<16} | {'[N=648]':<16} | {'[N=720]':<14} | {'[N=36]'}")
    print(f"-" * 95)
    for cname, code in [("Static", "S"), ("Trust-Agnostic", "TA"), ("Trust-Calibrated", "TC")]:
        m = metrics_map[cname]
        acc_str = f"{m['accuracy']*100:.1f} ({m['accuracy_raw']})"
        abr_str = f"{m['abr']*100:.1f} ({m['abr_raw']})"
        irr_str = f"{m['irr']*100:.1f} ({m['irr_raw']})"
        time_str = f"{m['mean_time']:.1f} +/- {m['std_time']:.1f}"
        tlx_str = f"{m['nasa_tlx']:.1f}"
        print(f"{code + ' (' + cname + ')':<18} | {acc_str:<16} | {abr_str:<16} | {irr_str:<16} | {time_str:<14} | {tlx_str:<10}")
    print(f"=======================================================\n")

    # 3. Fit Mixed-Effects Regressions (Table 7 in paper)
    print(f"[*] Fitting Hierarchical Crossed Mixed-Effects Regressions...")
    glmm_acc = CrossedMixedEffectsRegressor.fit_glmm_logistic(trials, outcome_key="is_correct_decision")
    
    # Filter subsets for reliance models
    err_subset = [t for t in trials if not t["is_ai_correct"]]
    glmm_abr = CrossedMixedEffectsRegressor.fit_glmm_logistic(err_subset, outcome_key="accepted_ai")

    corr_subset = [t for t in trials if t["is_ai_correct"]]
    # Inappropriate rejection: outcome is NOT accepted
    for t in corr_subset:
        t["is_inappropriate_rejection"] = not t["accepted_ai"]
    glmm_irr = CrossedMixedEffectsRegressor.fit_glmm_logistic(corr_subset, outcome_key="is_inappropriate_rejection")

    lmm_time = CrossedMixedEffectsRegressor.fit_lmm_continuous(trials, outcome_key="response_time")

    print(f"\n=======================================================")
    print(f" TABLE: Mixed-Effects Regression Models (Summary Contrasts)")
    print(f"=======================================================")
    print(f"{'Outcome Model':<22} | {'Contrast':<12} | {'Beta / Coef':<12} | {'z / t':<8} | {'p_adj':<8} | {'Effect Size [95% CI]'}")
    print(f"-" * 90)

    # Accuracy
    tc_acc = glmm_acc["parameters"]["TC vs S"]
    print(f"{'Decision Accuracy':<22} | {'TC vs S':<12} | {tc_acc['beta']:+0.2f}         | {tc_acc['z']:+0.2f}   | {tc_acc['p_adj']:.4f} | OR {tc_acc['odds_ratio']:.2f} [{tc_acc['or_ci'][0]:.2f}, {tc_acc['or_ci'][1]:.2f}]")
    
    # ABR
    tc_abr = glmm_abr["parameters"]["TC vs S"]
    print(f"{'Automation Bias (ABR)':<22} | {'TC vs S':<12} | {tc_abr['beta']:+0.2f}         | {tc_abr['z']:+0.2f}   | {tc_abr['p_adj']:.4f} | OR {tc_abr['odds_ratio']:.2f} [{tc_abr['or_ci'][0]:.2f}, {tc_abr['or_ci'][1]:.2f}]")

    # IRR
    tc_irr = glmm_irr["parameters"]["TC vs S"]
    print(f"{'Inapp. Rejection (IRR)':<22} | {'TC vs S':<12} | {tc_irr['beta']:+0.2f}         | {tc_irr['z']:+0.2f}   | {tc_irr['p_adj']:.4f} | OR {tc_irr['odds_ratio']:.2f} [{tc_irr['or_ci'][0]:.2f}, {tc_irr['or_ci'][1]:.2f}]")

    # Response Time
    tc_time = lmm_time["parameters"]["TC vs S"]
    print(f"{'Response Time (s)':<22} | {'TC vs S':<12} | {tc_time['coef']:+0.2f} s       | {tc_time['t']:+0.2f}   | {tc_time['p_adj']:.4f} | Diff {tc_time['coef']:.1f}s [{tc_time['ci_diff'][0]:.1f}, {tc_time['ci_diff'][1]:.1f}]")
    print(f"=======================================================\n")

    # 4. Save Artifacts & Figures
    out_dir = Path(args.output_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_operational_comparison(metrics_map, save_path=str(fig_dir / "operational_comparison.png"))
    plot_state_trajectory(trials, participant_id=1, save_path=str(fig_dir / "state_trajectory_participant_1.png"))
    plot_explanation_distribution(trials, save_path=str(fig_dir / "explanation_level_distribution.png"))

    summary_file = out_dir / "experiment_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": metrics_map,
            "glmm_accuracy": glmm_acc,
            "glmm_abr": glmm_abr,
            "glmm_irr": glmm_irr,
            "lmm_response_time": lmm_time,
        }, f, indent=2)

    print(f"[*] Figures generated in: {fig_dir}")
    print(f"[*] Full experimental summary saved to: {summary_file}\n")


if __name__ == "__main__":
    main()
