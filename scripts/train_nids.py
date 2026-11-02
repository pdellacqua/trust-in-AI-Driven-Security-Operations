#!/usr/bin/env python3
"""
Training and Reliability Calibration Pipeline for Contrastive Multimodal NIDS.
Usage:
    python3 scripts/train_nids.py --dataset CSE-CIC-IDS2018 --epochs 15
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.nids.synthetic_data import create_benchmark_splits
from src.nids.dataset import FlowDataset
from src.nids.contrastive_model import MultimodalNIDS
from src.nids.calibration import (
    TemperatureScaler,
    compute_ece,
    compute_reliability_diagram_data,
    plot_reliability_diagram,
)
from src.nids.baselines import TrafficOnlyMLP, BaselineRandomForest, mcnemar_test


def parse_args():
    parser = argparse.ArgumentParser(description="Train and Calibrate Multimodal Traffic-Text NIDS")
    parser.add_argument("--dataset", type=str, default="CSE-CIC-IDS2018", choices=["CSE-CIC-IDS2018", "UNSW-NB15", "TON_IoT"])
    parser.add_argument("--samples", type=int, default=8000, help="Total synthetic flow samples")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results/nids")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n=======================================================")
    print(f" Multimodal Traffic-Text NIDS Training Pipeline")
    print(f" Dataset: {args.dataset} | Device: {device}")
    print(f"=======================================================\n")

    # 1. Dataset Partitioning (with held-out zero-shot classes)
    splits = create_benchmark_splits(dataset_name=args.dataset, total_samples=args.samples, seed=args.seed)
    num_classes = len(splits["classes"])

    train_ds = FlowDataset(splits["train"]["X"], splits["train"]["y"])
    val_ds = FlowDataset(splits["val"]["X"], splits["val"]["y"])
    test_ds = FlowDataset(splits["test_closed"]["X"], splits["test_closed"]["y"])
    zero_ds = FlowDataset(splits["test_zero_shot"]["X"], splits["test_zero_shot"]["y"])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    zero_loader = DataLoader(zero_ds, batch_size=args.batch_size, shuffle=False)

    print(f"[*] Training samples: {len(train_ds)} | Val: {len(val_ds)} | Closed Test: {len(test_ds)}")
    print(f"[*] Held-out zero-shot classes: {splits['held_out_classes']} ({len(zero_ds)} samples)")

    # 2. Model Initialization
    model = MultimodalNIDS(
        input_dim=splits["train"]["X"].shape[1],
        hidden_dim=128,
        embedding_dim=64,
        num_classes=num_classes,
        temperature=0.07,
        omega_loss=0.60,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # 3. Training Loop
    print("\n[*] Starting contrastive training...")
    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        n_batches = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            loss, metrics = model.compute_loss(x_batch, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"  Epoch [{epoch:02d}/{args.epochs:02d}] -> Total Loss: {avg_loss:.4f}")

    # 4. Evaluation and Reliability Calibration via Temperature Scaling
    print("\n[*] Optimizing Temperature Scaling Calibration on Validation Set...")
    model.eval()
    val_logits_list = []
    val_labels_list = []
    with torch.no_grad():
        for x_v, y_v in val_loader:
            x_v = x_v.to(device)
            _, logits = model(x_v)
            val_logits_list.append(logits.cpu())
            val_labels_list.append(y_v)

    val_logits = torch.cat(val_logits_list, dim=0)
    val_labels = torch.cat(val_labels_list, dim=0)

    scaler = TemperatureScaler(initial_temp=1.0)
    best_tau = scaler.fit(val_logits, val_labels)
    print(f"[*] Optimal Temperature Parameter (tau_cal): {best_tau:.3f}")

    # 5. Closed-Set Testing
    print("\n[*] Evaluating on Closed-Set Test Partition...")
    test_logits_list = []
    test_labels_list = []
    with torch.no_grad():
        for x_t, y_t in test_loader:
            x_t = x_t.to(device)
            _, logits = model(x_t)
            test_logits_list.append(logits.cpu())
            test_labels_list.append(y_t)

    test_logits = torch.cat(test_logits_list, dim=0)
    test_labels = torch.cat(test_labels_list, dim=0).numpy()

    uncal_probs = torch.softmax(test_logits, dim=-1).numpy()
    cal_probs = torch.softmax(test_logits / best_tau, dim=-1).numpy()

    y_pred_uncal = np.argmax(uncal_probs, axis=-1)
    y_pred_cal = np.argmax(cal_probs, axis=-1)
    accuracy = float(np.mean(y_pred_cal == test_labels))

    # Compute ECE pre and post calibration
    diag_data = compute_reliability_diagram_data(uncal_probs, cal_probs, test_labels, n_bins=10)
    ece_raw = diag_data["ece_raw"]
    ece_cal = diag_data["ece_cal"]

    print(f"\n=======================================================")
    print(f" Results Summary for {args.dataset}")
    print(f"=======================================================")
    print(f"  Accuracy:             {accuracy * 100.0:.2f}%")
    print(f"  Raw ECE (ECE_raw):    {ece_raw:.4f}")
    print(f"  Calibrated ECE:       {ece_cal:.4f}  (Reduced by {((ece_raw - ece_cal)/ece_raw)*100:.1f}%)")
    print(f"=======================================================\n")

    # 6. Save Reliability Diagram Plot
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"reliability_diagram_{args.dataset.lower()}.png"
    plot_reliability_diagram(diag_data, dataset_name=args.dataset, save_path=str(plot_path))
    print(f"[*] Reliability diagram saved to: {plot_path}")

    # 7. Baseline Model Comparison & McNemar Test
    print("\n[*] Evaluating Baseline Models and Paired McNemar Test...")
    rf = BaselineRandomForest(n_estimators=30, max_depth=10, random_state=args.seed)
    X_train_np = splits["train"]["X"]
    y_train_np = splits["train"]["y"]
    X_test_np = splits["test_closed"]["X"]
    rf.fit(X_train_np, y_train_np)
    y_pred_rf = rf.predict(X_test_np)

    rf_acc = float(np.mean(y_pred_rf == test_labels))
    chi2_val, p_val, counts = mcnemar_test(test_labels, y_pred_cal, y_pred_rf)
    print(f"  Random Forest Baseline Accuracy: {rf_acc * 100.0:.2f}%")
    print(f"  McNemar Test vs Proposed NIDS: chi2 = {chi2_val:.2f}, p-value = {p_val:.4e}")
    print(f"  Contingency: b (NIDS correct, RF wrong) = {counts['b']}, c (RF correct, NIDS wrong) = {counts['c']}")


if __name__ == "__main__":
    main()
