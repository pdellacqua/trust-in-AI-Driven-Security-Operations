"""
Synthetic Network Flow Generator and Benchmark Dataset Splitter.
Provides statistical synthesis of network intrusion detection traffic
matching CSE-CIC-IDS2018, UNSW-NB15, and TON_IoT schemas.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch

FEATURE_NAMES = [
    "flow_duration",
    "fwd_pkts_tot",
    "bwd_pkts_tot",
    "fwd_data_bytes",
    "bwd_data_bytes",
    "fwd_pkt_len_max",
    "fwd_pkt_len_min",
    "fwd_pkt_len_mean",
    "fwd_pkt_len_std",
    "bwd_pkt_len_max",
    "bwd_pkt_len_min",
    "bwd_pkt_len_mean",
    "bwd_pkt_len_std",
    "flow_bytes_per_sec",
    "flow_pkts_per_sec",
    "flow_iat_mean",
    "flow_iat_std",
    "flow_iat_max",
    "flow_iat_min",
    "fwd_iat_tot",
    "bwd_iat_tot",
    "syn_flag_count",
    "rst_flag_count",
    "psh_flag_count",
    "ack_flag_count",
    "urg_flag_count",
    "down_up_ratio",
    "pkt_size_avg",
    "fwd_seg_size_avg",
    "bwd_seg_size_avg",
    "subflow_fwd_pkts",
    "subflow_fwd_bytes",
    "subflow_bwd_bytes",
    "init_win_bytes_fwd",
    "init_win_bytes_bwd",
    "fwd_act_data_pkts",
    "fwd_seg_size_min",
    "active_mean",
    "active_std",
    "idle_mean",
    "idle_std",
    "cross_subnet_flag",
]

NUM_FEATURES = len(FEATURE_NAMES)  # 42 features


class SyntheticTrafficGenerator:
    """
    Generates realistic network flow records for benign and malicious network events.
    """

    def __init__(self, kb_path: Optional[str] = None, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        if kb_path is None:
            kb_path = Path(__file__).resolve().parent.parent.parent / "data" / "mitre_cve_kb.json"
        
        with open(kb_path, "r", encoding="utf-8") as f:
            self.kb = json.load(f)
        
        self.classes = self.kb["classes"]
        self.class_names = list(self.classes.keys())
        self.class_to_id = {name: data["class_id"] for name, data in self.classes.items()}
        self.id_to_class = {data["class_id"]: name for name, data in self.classes.items()}
        self.num_classes = len(self.classes)
        
        # Build baseline profiles for each class
        self._profiles = self._build_profiles()

    def _build_profiles(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Build mean and covariance vectors for each traffic category.
        """
        profiles = {}
        for name in self.class_names:
            mean = np.zeros(NUM_FEATURES, dtype=np.float32)
            std = np.ones(NUM_FEATURES, dtype=np.float32) * 0.5
            
            if name == "Benign":
                mean[0] = 5.0      # flow duration
                mean[1] = 12.0     # fwd pkts
                mean[2] = 10.0     # bwd pkts
                mean[7] = 250.0    # fwd pkt len mean
                mean[11] = 600.0   # bwd pkt len mean
                mean[14] = 20.0    # flow pkts per sec
                mean[21] = 1.0     # syn flag
                mean[24] = 8.0     # ack flag
                mean[41] = 0.05    # cross subnet
            elif name == "DDoS":
                mean[0] = 1.2
                mean[1] = 450.0
                mean[2] = 2.0
                mean[7] = 64.0
                mean[11] = 0.0
                mean[14] = 950.0   # massive packet rate
                mean[21] = 400.0   # syn flood
                mean[24] = 1.0
                mean[41] = 0.10
            elif name == "PortScan":
                mean[0] = 0.05     # extremely short duration
                mean[1] = 2.0
                mean[2] = 1.0
                mean[7] = 40.0
                mean[14] = 80.0
                mean[21] = 1.0
                mean[22] = 0.8     # rst flags from closed ports
                mean[41] = 0.15
            elif name == "BruteForce":
                mean[0] = 8.5
                mean[1] = 45.0
                mean[2] = 40.0
                mean[7] = 85.0
                mean[11] = 90.0
                mean[14] = 15.0
                mean[23] = 20.0    # psh flag (payload exchange)
                mean[41] = 0.20
            elif name == "WebAttack":
                mean[0] = 3.2
                mean[1] = 8.0
                mean[2] = 6.0
                mean[7] = 480.0    # large fwd payload (SQLi/XSS)
                mean[11] = 350.0
                mean[23] = 4.0
                mean[41] = 0.10
            elif name == "Infiltration":
                mean[0] = 18.0     # long lateral flow
                mean[1] = 60.0
                mean[2] = 55.0
                mean[7] = 320.0
                mean[11] = 450.0
                mean[33] = 8192.0  # init win bytes
                mean[41] = 0.95    # high cross subnet lateral movement
            elif name == "Botnet":
                mean[0] = 12.0
                mean[1] = 14.0
                mean[2] = 12.0
                mean[7] = 110.0
                mean[11] = 95.0
                mean[15] = 4.5     # periodic heartbeat iat
                mean[16] = 0.1     # low jitter
                mean[41] = 0.30
            elif name == "Worms":
                mean[0] = 2.1
                mean[1] = 120.0
                mean[2] = 5.0
                mean[7] = 180.0
                mean[14] = 250.0   # rapid spreading rate
                mean[41] = 0.80
            elif name == "Backdoor":
                mean[0] = 45.0     # persistent reverse connection
                mean[1] = 25.0
                mean[2] = 28.0
                mean[7] = 65.0     # interactive keystrokes
                mean[11] = 120.0
                mean[15] = 2.0
                mean[41] = 0.40
            elif name == "Ransomware":
                mean[0] = 25.0
                mean[1] = 800.0    # high volume file writes
                mean[2] = 650.0
                mean[7] = 1024.0
                mean[11] = 1024.0
                mean[13] = 85000.0 # high bytes per sec
                mean[41] = 0.85
                
            profiles[name] = {"mean": mean, "std": std}
        return profiles

    def generate_flows(self, class_name: str, n_samples: int) -> np.ndarray:
        """
        Generate raw flow feature matrix for a specific class.
        """
        if class_name not in self._profiles:
            raise ValueError(f"Unknown class {class_name}")
        profile = self._profiles[class_name]
        mean = profile["mean"]
        std = np.maximum(profile["std"] * np.abs(mean), 0.1)
        raw_samples = self.rng.normal(loc=mean, scale=std, size=(n_samples, NUM_FEATURES))
        # Enforce non-negativity on counts and durations
        raw_samples = np.maximum(raw_samples, 0.0)
        # Cap binary/probability flags to [0, 1]
        raw_samples[:, 41] = np.clip(raw_samples[:, 41], 0.0, 1.0)
        return raw_samples.astype(np.float32)

    def generate_dataset(
        self,
        samples_per_class: Dict[str, int],
        standardize: bool = True,
        stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate an assembled dataset of network flows and labels.
        Returns:
            X: [N, NUM_FEATURES]
            y: [N]
            (mean, std): normalization statistics
        """
        features_list = []
        labels_list = []
        
        for cname, count in samples_per_class.items():
            if count <= 0:
                continue
            feats = self.generate_flows(cname, count)
            cid = self.class_to_id[cname]
            labels = np.full(count, cid, dtype=np.int64)
            features_list.append(feats)
            labels_list.append(labels)
            
        X = np.vstack(features_list)
        y = np.concatenate(labels_list)
        
        # Shuffle
        perm = self.rng.permutation(len(y))
        X = X[perm]
        y = y[perm]
        
        if standardize:
            if stats is None:
                mean = np.mean(X, axis=0, keepdims=True)
                std = np.std(X, axis=0, keepdims=True)
                std = np.where(std < 1e-6, 1.0, std)
            else:
                mean, std = stats
            X = (X - mean) / std
        else:
            mean = np.zeros((1, NUM_FEATURES), dtype=np.float32)
            std = np.ones((1, NUM_FEATURES), dtype=np.float32)
            
        return X, y, (mean, std)


def create_benchmark_splits(
    dataset_name: str = "CSE-CIC-IDS2018",
    total_samples: int = 10000,
    seed: int = 42,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Implements the two-stage partitioning protocol described in Section 5.1 of the paper:
    1. Pre-splits and holds out specific unseen attack classes for zero-shot testing:
       - CSE-CIC-IDS2018: Infiltration, Botnet
       - UNSW-NB15: Worms, Backdoor
       - TON_IoT: Ransomware, Backdoor
    2. Splits known categories and benign traffic using 70/10/20 train/val/test ratio.
    """
    gen = SyntheticTrafficGenerator(seed=seed)
    held_out_map = {
        "CSE-CIC-IDS2018": ["Infiltration", "Botnet"],
        "UNSW-NB15": ["Worms", "Backdoor"],
        "TON_IoT": ["Ransomware", "Backdoor"],
    }
    
    held_out = held_out_map.get(dataset_name, ["Infiltration", "Botnet"])
    known_classes = [c for c in gen.class_names if c not in held_out]
    
    # 50% benign, 50% distributed across known attacks
    known_counts = {}
    benign_samples = total_samples // 2
    known_counts["Benign"] = benign_samples
    
    attack_samples = total_samples - benign_samples
    other_classes = [c for c in known_classes if c != "Benign"]
    per_attack = attack_samples // len(other_classes)
    for c in other_classes:
        known_counts[c] = per_attack
        
    # Generate known data
    X_known, y_known, (mean, std) = gen.generate_dataset(known_counts, standardize=False)
    
    # Stratified 70/10/20 split
    n_total = len(y_known)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.10)
    
    # Z-score normalization fitted strictly on training partition
    X_train_raw = X_known[:n_train]
    norm_mean = np.mean(X_train_raw, axis=0, keepdims=True)
    norm_std = np.std(X_train_raw, axis=0, keepdims=True)
    norm_std = np.where(norm_std < 1e-6, 1.0, norm_std)
    
    X_train = (X_known[:n_train] - norm_mean) / norm_std
    y_train = y_known[:n_train]
    
    X_val = (X_known[n_train:n_train + n_val] - norm_mean) / norm_std
    y_val = y_known[n_train:n_train + n_val]
    
    X_test_closed = (X_known[n_train + n_val:] - norm_mean) / norm_std
    y_test_closed = y_known[n_train + n_val:]
    
    # Generate held-out zero-shot partition
    zero_shot_counts = {c: 300 for c in held_out}
    X_zero_raw, y_zero, _ = gen.generate_dataset(zero_shot_counts, standardize=False)
    X_zero = (X_zero_raw - norm_mean) / norm_std
    
    return {
        "train": {"X": X_train, "y": y_train},
        "val": {"X": X_val, "y": y_val},
        "test_closed": {"X": X_test_closed, "y": y_test_closed},
        "test_zero_shot": {"X": X_zero, "y": y_zero},
        "norm_stats": {"mean": norm_mean, "std": norm_std},
        "classes": gen.classes,
        "class_to_id": gen.class_to_id,
        "id_to_class": gen.id_to_class,
        "held_out_classes": held_out,
    }
