## Overview

This repo builds a full pipeline for **graph signal prediction & reconstruction** on time-varying UAV communication graphs:

1) **Data creation** — simulate UAV trajectories, scale to [0,1], compute SNR, build per-timestep graphs → PyTorch Geometric `Data` list.  
2) **Graph Autoencoder (GAE)** — learn low-dimensional graph embeddings and reconstruct node signals.  
3) **Koopman Autoencoder (KAE)** — learn linear dynamics in latent space and forecast embeddings.
4) **Reconstruction** — Recover full node states from the forecasted latents, even with masked/missing nodes.

> License: This repo includes files licensed under **GPL-3.0**. See `LICENSE` and file headers for details.

---

## Quickstart

### 0) Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
### 1) Create the UAV dataset

This step runs the simulation, preprocessing, and graph building.


```bash
python -m src.uavnet.make_dataset --outdir data/processed/run1
```

### 2) Train the Graph Autoencoder (GAE)

Trains a small encoder (SAGEConv → latent=20 → global mean pool) + MLP decoder to reconstruct node signals.

```bash
python -m src.uavnet.train_gae \
  --data_dir data/processed/run1 \
  --epochs 200 \
  --device cpu
```

Evaluate GAE reconstruction

```bash
python -m src.uavnet.eval_gae --data_dir data/processed/run1
```

### 3) Koopman Autoencoder (latent forecasting)

#### 3.1 Preprocess for KAE (your notebook logic, separated)

This turns the saved GAE latents into tensors for the Koopman model using your original steps (min–max scaling, channel add, TL windows).

```bash
python -m src.uavnet.koopman_preproc_user \
  --data_dir data/processed/run1 \
  --TL 50 \
  --num_uavs 20 \
  --bottle 8
```

#### 3.2 Train the KAE on graph embeddings

```bash
python -m src.uavnet.koopman_train_user \
  --data_dir data/processed/run1 \
  --epochs 400 \
  --device cpu \
  --lr 1e-2 \
  --batch_size 16 \
  --learning_rate_change 0.2 \
  --epoch_update 300 350 \
  --backward 0
```

#### 3.3 Evaluate GKAE predictions

```bash
python -m src.uavnet.koopman_eval_user \
  --data_dir data/processed/run1 \
  --pred_length 1 \
  --device cpu
```

### 4) Reconstruction Task

#### 4.1 Create Reconstruction Dataset

```bash
python -m src.uavnet.make_recon_dataset \
  --data_dir data/processed/run1 \
  --mask_ratio 0.4
```
#### 4.2 Train Reconstruction Model

```bash
python -m src.uavnet.train_recon \
  --data_dir data/processed/run1 \
  --epochs 500 \
  --lr 0.01 \
  --device cpu
```

#### 4.3 Evaluate Reconstruction

```bash
python -m src.uavnet.eval_recon --data_dir data/processed/run1
```

