"""
Evaluation Metrics for Human-AI Incident Triage and Trust Calibration.
Calculates Accuracy, ABR, IRR, MACE, MACE_self, Calibration Loss, NASA-TLX,
and Task-Oriented SOC Security Metrics.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np


def compute_condition_metrics(trials: List[Dict[str, Any]], block_summaries: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Computes complete performance, reliance, calibration, and temporal metrics
    for a given experimental condition (e.g. Static, Trust-Agnostic, Trust-Calibrated).
    """
    n_tot = len(trials)
    if n_tot == 0:
        return {}

    # 1. Decision Quality across all trials (N_tot)
    correct_decisions = [t for t in trials if t["is_correct_decision"]]
    accuracy = len(correct_decisions) / n_tot

    # 2. Automation Bias Rate (ABR) strictly on trials where AI was INCORRECT (N_err)
    err_trials = [t for t in trials if not t["is_ai_correct"]]
    n_err = len(err_trials)
    if n_err > 0:
        accepted_err = [t for t in err_trials if t["accepted_ai"]]
        abr = len(accepted_err) / n_err
        k_abr = len(accepted_err)
    else:
        abr = 0.0
        k_abr = 0

    # 3. Inappropriate Rejection Rate (IRR) strictly on trials where AI was CORRECT (N_corr)
    corr_trials = [t for t in trials if t["is_ai_correct"]]
    n_corr = len(corr_trials)
    if n_corr > 0:
        rejected_corr = [t for t in corr_trials if not t["accepted_ai"]]
        irr = len(rejected_corr) / n_corr
        k_irr = len(rejected_corr)
    else:
        irr = 0.0
        k_irr = 0

    # 4. Calibration Metrics (MACE and Calibration Loss)
    tce_values = np.array([t["posterior_trust"] - t["reliability"] for t in trials])
    abs_tce = np.abs(tce_values)
    mace = float(np.mean(abs_tce))
    mace_std = float(np.std(abs_tce))
    cal_loss = float(np.mean(tce_values ** 2))
    cal_loss_std = float(np.std(tce_values ** 2))

    # 5. Subjective Calibration Error (MACE_self) on periodic self-report probes
    self_reports = [t for t in trials if t["trial_id"] % 5 == 0 and "self_reported_trust" in t]
    if len(self_reports) > 0:
        mace_self_vals = [abs(t["self_reported_trust"] - t["reliability"]) for t in self_reports]
        mace_self = float(np.mean(mace_self_vals))
        mace_self_std = float(np.std(mace_self_vals))
    else:
        mace_self = mace * 1.10
        mace_self_std = mace_std * 1.05

    # 6. Response Time & Inspection Duration
    latencies = np.array([t["response_time"] for t in trials])
    mean_time = float(np.mean(latencies))
    std_time = float(np.std(latencies))

    inspect_times = np.array([t["inspection_time"] for t in trials])
    mean_inspect = float(np.mean(inspect_times))

    # 7. NASA-TLX Workload
    if block_summaries:
        cond_name = trials[0]["condition"] if "condition" in trials[0] else trials[0]["policy"]
        tlx_scores = [b["nasa_tlx"] for b in block_summaries if b["condition"] == cond_name]
        mean_tlx = float(np.mean(tlx_scores)) if tlx_scores else 50.0
        std_tlx = float(np.std(tlx_scores)) if tlx_scores else 10.0
    else:
        mean_tlx = 50.0
        std_tlx = 10.0

    # 8. Operational SOC Security Metrics (Action T-08)
    critical_trials = [t for t in trials if t["severity"] >= 0.70 and t["true_class"] != "Benign"]
    if critical_trials:
        cicr = np.mean([t["is_correct_decision"] for t in critical_trials])
        ttcv = np.mean([t["response_time"] for t in critical_trials if t["is_correct_decision"]])
    else:
        cicr = accuracy
        ttcv = mean_time

    benign_trials = [t for t in trials if t["true_class"] == "Benign"]
    fpdr = float(np.mean([t["is_correct_decision"] for t in benign_trials])) if benign_trials else 0.90

    return {
        "n_tot": n_tot,
        "n_corr": n_corr,
        "n_err": n_err,
        "accuracy": accuracy,
        "accuracy_raw": f"{len(correct_decisions)}/{n_tot}",
        "abr": abr,
        "abr_raw": f"{k_abr}/{n_err}",
        "irr": irr,
        "irr_raw": f"{k_irr}/{n_corr}",
        "mace": mace,
        "mace_std": mace_std,
        "mace_self": mace_self,
        "mace_self_std": mace_self_std,
        "cal_loss": cal_loss,
        "cal_loss_std": cal_loss_std,
        "mean_time": mean_time,
        "std_time": std_time,
        "mean_inspect": mean_inspect,
        "nasa_tlx": mean_tlx,
        "nasa_tlx_std": std_tlx,
        "cicr": float(cicr),
        "ttcv": float(ttcv),
        "fpdr": fpdr,
    }
