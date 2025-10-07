## Overview

This repo builds a full pipeline for **graph signal prediction & reconstruction** on time-varying UAV communication graphs:

1) **Data creation** — simulate UAV trajectories, scale to [0,1], compute SNR, build per-timestep graphs → PyTorch Geometric `Data` list.  
2) **Graph Autoencoder (GAE)** — learn low-dimensional graph embeddings and reconstruct node signals.  
3) **Koopman Autoencoder (KAE)** — learn linear dynamics in latent space and forecast embeddings.
4) **Reconstruction** — Reconstruct the future embeddings with masked nodes (to be added). 

> License: This repo includes files licensed under **GPL-3.0**. See `LICENSE` and file headers for details.

---

## Quickstart

### 0) Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
