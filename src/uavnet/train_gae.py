import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from .models import Autoencoder, NUM_UAVS

def build_loaders(pyg_graphs_path: Path, batch_size: int = 1):
    pyg_list = torch.load(pyg_graphs_path, weights_only=False)
    train_loader = DataLoader(pyg_list[:300], batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(pyg_list[:500], batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def train_one_epoch(model, loader, opt, device):
    model.train()
    cum_loss = 0.0
    opt.zero_grad()
    for data in loader:
        data = data.to(device)
        x_recon, _ = model(data)                # [B, N, 1]
        # reshape to match data.x [total_nodes, 1] for B=1
        x_recon_flat = x_recon.view(-1, data.num_features)
        loss = F.mse_loss(data.x, x_recon_flat)
        loss.backward()
        opt.step()
        opt.zero_grad()
        cum_loss += loss.item()
    return cum_loss / max(1, len(loader))

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    cum_loss = 0.0
    if len(loader) == 0:
        return float("inf")
    for data in loader:
        data = data.to(device)
        x_recon, _ = model(data)
        x_recon_flat = x_recon.view(-1, data.num_features)
        loss = F.mse_loss(data.x, x_recon_flat)
        cum_loss += loss.item()
    return cum_loss / len(loader)

@torch.no_grad()
def dump_val_predictions_and_latents(model, loader, device):
    """
    Returns:
      real:  (T, N) numpy
      pred:  (T, N) numpy
      z_all: (T, latent_dim) numpy
    where T = number of graphs in 'loader' (500 in your split) and N=NUM_UAVS
    """
    model.eval()
    real_list, pred_list, z_list = [], [], []
    for data in loader:
        data = data.to(device)
        x_recon, z = model(data)                      # x_recon: [B,N,1], z: [B,latent]
        x_recon_flat = x_recon.view(-1, data.num_features)  # [N,1] for B=1
        real_list.append(data.x.detach().cpu().numpy().reshape(NUM_UAVS))
        pred_list.append(x_recon_flat.detach().cpu().numpy().reshape(NUM_UAVS))
        z_list.append(z.detach().cpu().numpy().reshape(-1))
    real = np.stack(real_list, axis=0)  # (T,N)
    pred = np.stack(pred_list, axis=0)  # (T,N)
    z_all = np.stack(z_list, axis=0)    # (T,latent)
    return real, pred, z_all

def minmax_scale(arr):
    lo, hi = arr.min(), arr.max()
    ptp = (hi - lo) if (hi > lo) else 1.0
    return (arr - lo) / ptp

def main():
    ap = argparse.ArgumentParser(description="Train simple graph autoencoder on PyG graphs")
    ap.add_argument("--data_dir", type=str, default="data/processed/run1", help="Folder containing pyg_graphs.pt")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--outdir", type=str, default=None, help="Where to save outputs; defaults to data_dir")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    pyg_path = data_dir / "pyg_graphs.pt"
    if not pyg_path.exists():
        raise FileNotFoundError(f"Cannot find {pyg_path}. Run the dataset builder first.")

    outdir = Path(args.outdir) if args.outdir else data_dir
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = Autoencoder(in_features=1, latent_dim=20, num_uavs=NUM_UAVS).to(device)

    train_loader, val_loader = build_loaders(pyg_path, batch_size=1)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.2, patience=30)

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, opt, device)
        va = validate(model, val_loader, device)
        sched.step(va)
        if epoch == 175:
            for pg in opt.param_groups:
                pg["lr"] *= 0.2
            print(f"Manually reduced LR at epoch {epoch} -> {opt.param_groups[0]['lr']:.3e}")
        print(f"Epoch {epoch:03d} | train {tr:.8f} | val {va:.8f}")

    # Dump predictions/latents on validation split
    real, pred, z = dump_val_predictions_and_latents(model, val_loader, device)
    # Scale latent with min-max (global)
    z_scaled = minmax_scale(z)

    # Train/Test splits as per your notebook
    X_train = z_scaled[0:300]
    X_test  = z_scaled[299:500]

    # Save artifacts
    np.savez_compressed(outdir / "gae_outputs.npz",
                        real=real, pred=pred, z=z, z_scaled=z_scaled,
                        X_train=X_train, X_test=X_test)

    torch.save(model.state_dict(), outdir / "gae_model.pt")
    print(f"Saved: {outdir/'gae_outputs.npz'}")
    print(f"Saved: {outdir/'gae_model.pt'}")

if __name__ == "__main__":
    main()
