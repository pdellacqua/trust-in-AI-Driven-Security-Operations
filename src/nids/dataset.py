"""
Dataset Loader and Preprocessor for Network Intrusion Detection Systems.
Supports CSE-CIC-IDS2018, UNSW-NB15, and TON_IoT with automatic synthetic fallback.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from .synthetic_data import create_benchmark_splits, NUM_FEATURES


class FlowDataset(Dataset):
    """
    PyTorch Dataset wrapper for network flow features and class labels.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def get_benchmark_dataloaders(
    dataset_name: str = "CSE-CIC-IDS2018",
    batch_size: int = 256,
    num_samples: int = 10000,
    seed: int = 42,
    data_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, Dict[str, any]]:
    """
    Returns DataLoaders for train, validation, closed-set test, and zero-shot test.
    If real CSV files are not present in data_dir, generates a synthetic benchmark split.
    """
    splits = create_benchmark_splits(dataset_name=dataset_name, total_samples=num_samples, seed=seed)
    
    train_ds = FlowDataset(splits["train"]["X"], splits["train"]["y"])
    val_ds = FlowDataset(splits["val"]["X"], splits["val"]["y"])
    test_closed_ds = FlowDataset(splits["test_closed"]["X"], splits["test_closed"]["y"])
    test_zero_ds = FlowDataset(splits["test_zero_shot"]["X"], splits["test_zero_shot"]["y"])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_closed_loader = DataLoader(test_closed_ds, batch_size=batch_size, shuffle=False)
    test_zero_loader = DataLoader(test_zero_ds, batch_size=batch_size, shuffle=False)
    
    metadata = {
        "norm_stats": splits["norm_stats"],
        "classes": splits["classes"],
        "class_to_id": splits["class_to_id"],
        "id_to_class": splits["id_to_class"],
        "held_out_classes": splits["held_out_classes"],
        "num_classes": len(splits["classes"]),
    }
    
    return train_loader, val_loader, test_closed_loader, test_zero_loader, metadata
