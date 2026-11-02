"""
Probability Calibration via Temperature Scaling and ECE Evaluation.
Implements:
- Temperature scaling optimization via NLL minimization on validation data
- Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
- Reliability diagram generation for raw and calibrated model outputs
"""

import os
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class TemperatureScaler(nn.Module):
    """
    Applies temperature scaling to neural network logits:
    p_{t,c} = exp(ell_{t,c} / tau_cal) / sum_j exp(ell_{t,j} / tau_cal)
    """

    def __init__(self, initial_temp: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * initial_temp)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Scales logits by 1 / tau_cal.
        """
        temp = torch.clamp(self.temperature, min=0.01, max=10.0)
        return logits / temp

    def fit(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> float:
        """
        Optimizes temperature parameter by minimizing Negative Log-Likelihood (NLL).
        """
        val_logits = val_logits.detach()
        val_labels = val_labels.detach()
        
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()

        def _eval():
            optimizer.zero_grad()
            scaled_logits = self.forward(val_logits)
            loss = criterion(scaled_logits, val_labels)
            loss.backward()
            return loss

        optimizer.step(_eval)
        # Ensure positive temperature
        with torch.no_grad():
            self.temperature.clamp_(min=0.05, max=10.0)
            
        return float(self.temperature.item())


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, float, Dict[str, np.ndarray]]:
    """
    Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE):
    ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|
    MCE = max_m |acc(B_m) - conf(B_m)|
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    confidences = np.max(probs, axis=-1)
    predictions = np.argmax(probs, axis=-1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0

    bin_accs = []
    bin_confs = []
    bin_counts = []

    n_samples = len(labels)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        count_in_bin = np.sum(in_bin)

        if count_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            diff = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += diff * (count_in_bin / n_samples)
            mce = max(mce, diff)
            bin_accs.append(accuracy_in_bin)
            bin_confs.append(avg_confidence_in_bin)
        else:
            bin_accs.append(0.0)
            bin_confs.append((bin_lower + bin_upper) / 2.0)
        bin_counts.append(count_in_bin)

    return float(ece), float(mce), {
        "bin_accs": np.array(bin_accs),
        "bin_confs": np.array(bin_confs),
        "bin_counts": np.array(bin_counts),
        "bin_lowers": bin_lowers,
        "bin_uppers": bin_uppers,
    }


def compute_reliability_diagram_data(
    uncalibrated_probs: np.ndarray,
    calibrated_probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, any]:
    """
    Computes calibration metrics before and after temperature scaling.
    """
    ece_raw, mce_raw, data_raw = compute_ece(uncalibrated_probs, labels, n_bins)
    ece_cal, mce_cal, data_cal = compute_ece(calibrated_probs, labels, n_bins)

    return {
        "ece_raw": ece_raw,
        "mce_raw": mce_raw,
        "data_raw": data_raw,
        "ece_cal": ece_cal,
        "mce_cal": mce_cal,
        "data_cal": data_cal,
    }


def plot_reliability_diagram(
    diagram_data: Dict[str, any],
    dataset_name: str = "CSE-CIC-IDS2018",
    save_path: Optional[str] = None,
) -> None:
    """
    Generates and saves a publication-quality reliability diagram comparing
    raw vs calibrated posterior probabilities.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    # Plot Raw
    ax = axes[0]
    raw = diagram_data["data_raw"]
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax.bar(
        raw["bin_confs"],
        raw["bin_accs"],
        width=0.08,
        alpha=0.6,
        color="#d9534f",
        edgecolor="black",
        label=f"Raw Model (ECE = {diagram_data['ece_raw']:.3f})",
    )
    ax.set_title(f"{dataset_name} - Raw Confidence")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Empirical Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")

    # Plot Calibrated
    ax = axes[1]
    cal = diagram_data["data_cal"]
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax.bar(
        cal["bin_confs"],
        cal["bin_accs"],
        width=0.08,
        alpha=0.6,
        color="#2b8a3e",
        edgecolor="black",
        label=f"Calibrated (ECE = {diagram_data['ece_cal']:.3f})",
    )
    ax.set_title(f"{dataset_name} - Temperature Scaled")
    ax.set_xlabel("Confidence")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()
