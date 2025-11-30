import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np

# Import your custom modules
from .recon_utils import GraphLatentDataset
from .recon_models import Autoencoder

def pointwise_mse_loss(output, target):
    return torch.mean((output - target) ** 2, dim=1).mean()

def pointwise_cosine_similarity(output, target):
    cos_sim = F.cosine_similarity(output, target, dim=1)
    return 1 - cos_sim.mean()

def train(model, loader, optimizer, device, accumulate_steps=32):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for i, (data, target_latent, target_recon) in enumerate(loader):
        data = data.to(device)
        target_latent = target_latent.to(device)
        target_recon = target_recon.to(device)

        latent, recon = model(data.x, data.edge_index, data.batch)

        # Ensure sizes match (handling potential shape mismatches)
        if target_recon.size() != recon.size():
             # Try simple transpose first
            if target_recon.size(0) == recon.size(1) and target_recon.size(1) == recon.size(0):
                target_recon = target_recon.transpose(0, 1)
            else:
                target_recon = target_recon.view_as(recon)

        # Calculate losses
        latent_loss = F.mse_loss(latent, target_latent)
        recon_loss = F.mse_loss(recon, target_recon)
        pointwise_loss = pointwise_mse_loss(latent, target_latent)
        cosine_loss = pointwise_cosine_similarity(latent, target_latent)

        # Your custom weights
        loss = (0 * latent_loss +
                1e-8 * pointwise_loss +
                1 * recon_loss +
                0 * cosine_loss)

        loss.backward()
        total_loss += loss.item()

        if (i + 1) % accumulate_steps == 0 or (i + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()

    return total_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for data, target_latent, target_recon in loader:
            data = data.to(device)
            target_latent = target_latent.to(device)
            target_recon = target_recon.to(device)

            latent, recon = model(data.x, data.edge_index, data.batch)

            if target_recon.size() != recon.size():
                if target_recon.size(0) == recon.size(1) and target_recon.size(1) == recon.size(0):
                    target_recon = target_recon.transpose(0, 1)
                else:
                    target_recon = target_recon.view_as(recon)

            latent_loss = F.mse_loss(latent, target_latent)
            recon_loss = F.mse_loss(recon, target_recon)
            pointwise_loss = pointwise_mse_loss(latent, target_latent)
            cosine_loss = pointwise_cosine_similarity(latent, target_latent)

            total_loss = (0 * latent_loss +
                          1e-8 * pointwise_loss +
                          1 * recon_loss +
                          0 * cosine_loss)
            
            total_val_loss += total_loss.item()
    return total_val_loss / len(loader)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/processed/run1")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--latent_dim", type=int, default=20) # 'b' in your code
    args = ap.parse_args()

    device = torch.device(args.device)
    data_dir = Path(args.data_dir)

    # 1. Load Dataset
    print(f"Loading dataset from {data_dir / 'recon_dataset.pt'}...")
    full_dataset = torch.load(data_dir / "recon_dataset.pt", weights_only=False)
    
    train_loader = DataLoader(full_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(full_dataset, batch_size=1, shuffle=False)

    # 2. Initialize Model
    num_features = full_dataset[0][0].x.shape[1]
    num_uavs = full_dataset[0][2].shape[0]

    model = Autoencoder(train_features=num_features, 
                        latent_dim=args.latent_dim, 
                        num_uavs=num_uavs).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # REMOVED verbose=True
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=100
    )

    # 3. Training Loop
    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)

        scheduler.step(val_loss)

        # Manual LR adjustment
        if epoch == 400:
            for pg in optimizer.param_groups:
                pg['lr'] *= 0.2
            print(f"Manually adjusted learning rate at epoch {epoch}")

        if epoch % 10 == 0:
            # Manually print LR since verbose is removed
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}')

    # 4. Save Model
    torch.save(model.state_dict(), data_dir / "recon_model.pt")
    print(f"Saved model to {data_dir / 'recon_model.pt'}")

if __name__ == "__main__":
    main()