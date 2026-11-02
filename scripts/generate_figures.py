#!/usr/bin/env python3
"""
Publication Figure and Diagram Generation Script.
Generates:
1. Reliability Diagrams pre/post Temperature Scaling
2. Operational Performance & Reliance Comparison (Acc, ABR, IRR, NASA-TLX)
3. Latent Trust and Cognitive Load State Trajectory
4. Explanation Level (E0 - E3) Empirical Distribution
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.closed_loop import ClosedLoopSimulation
from src.evaluation.metrics import compute_condition_metrics
from src.evaluation.visualization import (
    plot_operational_comparison,
    plot_state_trajectory,
    plot_explanation_distribution,
)


def main():
    out_dir = PROJECT_ROOT / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[*] Generating publication figures in: {out_dir}")

    # Run lightweight cohort simulation to generate plotting data
    sim = ClosedLoopSimulation()
    cohort = sim.run_full_cohort(num_participants=12)
    trials = cohort["trials"]
    blocks = cohort["blocks"]

    trials_s = [t for t in trials if t["condition"] == "Static"]
    trials_ta = [t for t in trials if t["condition"] == "Trust-Agnostic"]
    trials_tc = [t for t in trials if t["condition"] == "Trust-Calibrated"]

    metrics_map = {
        "Static": compute_condition_metrics(trials_s, blocks),
        "Trust-Agnostic": compute_condition_metrics(trials_ta, blocks),
        "Trust-Calibrated": compute_condition_metrics(trials_tc, blocks),
    }

    # 1. Operational Comparisons
    p1 = out_dir / "fig3_operational_performance.png"
    plot_operational_comparison(metrics_map, save_path=str(p1))
    print(f"  [+] Saved {p1.name}")

    # 2. State Trajectory
    p2 = out_dir / "fig4_state_trajectory.png"
    plot_state_trajectory(trials, participant_id=1, save_path=str(p2))
    print(f"  [+] Saved {p2.name}")

    # 3. Explanation Level Distribution
    p3 = out_dir / "fig5_explanation_distribution.png"
    plot_explanation_distribution(trials, save_path=str(p3))
    print(f"  [+] Saved {p3.name}")

    print("[*] All publication figures successfully generated.\n")


if __name__ == "__main__":
    main()
