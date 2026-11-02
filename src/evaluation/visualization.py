"""
Visualization Utilities for Human-AI Incident Triage and Trust Calibration.
Generates publication-quality figures:
- Figure: Operational performance & reliance radar / bar comparisons
- Figure: Dynamic Trust, Reliability, and Cognitive Load trajectories
- Figure: Explanation level distribution across adaptive conditions (E0 - E3)
"""

import os
from typing import Dict, List, Any, Optional
import numpy as np
import matplotlib.pyplot as plt


def plot_operational_comparison(
    metrics_by_cond: Dict[str, Dict[str, Any]],
    save_path: Optional[str] = None,
) -> None:
    """
    Plots comparative bar charts of primary operational metrics across S, TA, and TC:
    - Decision Accuracy (%)
    - Automation Bias Rate (%)
    - Inappropriate Rejection Rate (%)
    - NASA-TLX Subjective Workload
    """
    conditions = ["Static", "Trust-Agnostic", "Trust-Calibrated"]
    labels = ["Static (S)", "Trust-Agnostic (TA)", "Trust-Calibrated (TC)"]
    colors = ["#7f7f7f", "#337ab7", "#2ca02c"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))

    # 1. Accuracy
    ax = axes[0]
    accs = [metrics_by_cond[c]["accuracy"] * 100.0 for c in conditions]
    bars = ax.bar(labels, accs, color=colors, width=0.55, edgecolor="black")
    ax.set_title("Decision Accuracy (%)", fontweight="bold")
    ax.set_ylim(70, 100)
    ax.grid(axis="y", linestyle=":", alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.8, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")

    # 2. Automation Bias Rate (ABR)
    ax = axes[1]
    abrs = [metrics_by_cond[c]["abr"] * 100.0 for c in conditions]
    bars = ax.bar(labels, abrs, color=colors, width=0.55, edgecolor="black")
    ax.set_title("Automation Bias Rate (%)", fontweight="bold")
    ax.set_ylim(0, 55)
    ax.grid(axis="y", linestyle=":", alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.0, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")

    # 3. Inappropriate Rejection Rate (IRR)
    ax = axes[2]
    irrs = [metrics_by_cond[c]["irr"] * 100.0 for c in conditions]
    bars = ax.bar(labels, irrs, color=colors, width=0.55, edgecolor="black")
    ax.set_title("Inappropriate Rejection (%)", fontweight="bold")
    ax.set_ylim(0, 25)
    ax.grid(axis="y", linestyle=":", alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.6, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")

    # 4. NASA-TLX Workload
    ax = axes[3]
    tlx = [metrics_by_cond[c]["nasa_tlx"] for c in conditions]
    bars = ax.bar(labels, tlx, color=colors, width=0.55, edgecolor="black")
    ax.set_title("NASA-TLX Workload [0-100]", fontweight="bold")
    ax.set_ylim(35, 70)
    ax.grid(axis="y", linestyle=":", alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.8, f"{yval:.1f}", ha="center", va="bottom", fontweight="bold")

    for ax in axes:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(["Static", "TA", "TC"], rotation=0)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_state_trajectory(
    trial_records: List[Dict[str, Any]],
    participant_id: int = 1,
    save_path: Optional[str] = None,
) -> None:
    """
    Plots dynamic temporal evolution of Latent Trust T_t, Model Reliability r_t,
    and Cognitive Load C_t over trials, highlighting perturbation points.
    """
    trials = [t for t in trial_records if t.get("participant_id") == participant_id]
    if not trials:
        trials = trial_records[:60]

    t_ids = [t["trial_id"] for t in trials]
    trusts = [t["posterior_trust"] for t in trials]
    reliabilities = [t["reliability"] for t in trials]
    loads = [t["posterior_load"] for t in trials]
    is_pert = [t["is_perturbation"] for t in trials]

    fig, ax1 = plt.subplots(figsize=(12, 4.8))

    ax1.plot(t_ids, trusts, color="#1f77b4", linewidth=2.0, marker="o", markersize=3, label="Inferred Trust (T_hat)")
    ax1.plot(t_ids, reliabilities, color="#ff7f0e", linestyle="--", linewidth=1.8, label="Model Reliability (r_t)")
    ax1.set_xlabel("Investigation Trial Index", fontweight="bold")
    ax1.set_ylabel("Trust & Reliability in [0, 1]", color="#1f77b4", fontweight="bold")
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Highlight error perturbations
    for i, pert in enumerate(is_pert):
        if pert:
            ax1.axvline(x=t_ids[i], color="#d62728", linestyle=":", alpha=0.8)
            ax1.scatter([t_ids[i]], [trusts[i]], color="#d62728", s=80, zorder=5)

    # Twin axis for cognitive load
    ax2 = ax1.twinx()
    ax2.plot(t_ids, loads, color="#2ca02c", linestyle="-.", linewidth=1.5, label="Cognitive Load (C_hat)")
    ax2.set_ylabel("Cognitive Workload in [0, 1]", color="#2ca02c", fontweight="bold")
    ax2.set_ylim(0.0, 1.05)

    # Combined legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title(f"Analyst State Trajectory (Participant #{participant_id}) - Closed-Loop Tracking", fontweight="bold")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_explanation_distribution(
    trials: List[Dict[str, Any]],
    save_path: Optional[str] = None,
) -> None:
    """
    Plots the percentage distribution of explanation actions (E0, E1, E2, E3)
    for Trust-Agnostic vs Trust-Calibrated conditions.
    """
    cond_ta = [t for t in trials if t.get("condition") in ["Trust-Agnostic", "TA"]]
    cond_tc = [t for t in trials if t.get("condition") in ["Trust-Calibrated", "TC"]]

    def get_dist(sub_trials):
        counts = {"E0": 0, "E1": 0, "E2": 0, "E3": 0}
        for t in sub_trials:
            act = t.get("action_name", "E1")
            counts[act] = counts.get(act, 0) + 1
        n = max(1, len(sub_trials))
        return [counts[k] / n * 100.0 for k in ["E0", "E1", "E2", "E3"]]

    dist_ta = get_dist(cond_ta)
    dist_tc = get_dist(cond_tc)

    labels = ["E0 (Minimal)", "E1 (Concise)", "E2 (Semantic)", "E3 (Full Investigation)"]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars1 = ax.bar(x - width / 2, dist_ta, width, label="Trust-Agnostic (TA)", color="#337ab7", edgecolor="black")
    bars2 = ax.bar(x + width / 2, dist_tc, width, label="Trust-Calibrated (TC)", color="#2ca02c", edgecolor="black")

    ax.set_ylabel("Selection Frequency (%)", fontweight="bold")
    ax.set_title("Empirical Distribution of Explanation Levels (E0 - E3)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_ylim(0, 60)
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.7)

    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.8, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.8, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()
