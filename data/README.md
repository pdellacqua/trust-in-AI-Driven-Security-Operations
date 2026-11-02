# Datasets Directory

This repository supports both **real cybersecurity benchmark datasets** and a **built-in high-fidelity synthetic benchmark generator**. As per standard research data practices and repository policies, raw gigabyte-scale datasets are **not** committed to version control.

## Supported Benchmark Datasets

If you have downloaded the official datasets, you can place their preprocessed CSV files in the respective subdirectories:

1. **CSE-CIC-IDS2018**
   - Source: Communications Security Establishment & Canadian Institute for Cybersecurity
   - Expected Directory: `data/cic-ids2018/`
   - Files: `02-14-2018.csv`, `02-15-2018.csv`, etc.
   - Zero-shot held-out classes: `Infiltration`, `Botnet Ares`

2. **UNSW-NB15**
   - Source: Cyber Range Lab of UNSW Canberra
   - Expected Directory: `data/unsw-nb15/`
   - Files: `UNSW-NB15_1.csv` ... `UNSW-NB15_4.csv`
   - Zero-shot held-out classes: `Worms`, `Backdoors`

3. **TON_IoT (Network Flow subset)**
   - Source: IoT Cybersecurity Lab, UNSW Canberra
   - Expected Directory: `data/ton_iot/`
   - Files: `Network_dataset.csv`
   - Zero-shot held-out classes: `Ransomware`, `Backdoor`

## Running Without Real Datasets (Zero Data Download)

You **do not need** to download any external datasets to run all training pipelines, EKF estimation, controller adaptation, experiments, and interactive SOC demonstrations. 

The framework includes a high-fidelity synthetic traffic generator (`src/nids/synthetic_data.py`) that generates statistically faithful flow representations matching the exact feature vectors, attack category distributions, and zero-shot partitions described in the paper.
