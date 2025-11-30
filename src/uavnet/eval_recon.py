import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from .recon_utils import GraphLatentDataset
from .recon_models import Autoencoder

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/processed/run1")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--start_idx", type=int, default=300, help="Start index matching make_recon_dataset")
    ap.add_argument("--end_idx", type=int, default=500, help="End index matching make_recon_dataset")
    ap.add_argument("--latent_dim", type=int, default=20)
    args = ap.parse_args()

    device = torch.device(args.device)
    data_dir = Path(args.data_dir)

    # 1. Load Data
    # Dataset (Masked graphs + Koopman predictions)
    dataset = torch.load(data_dir / "recon_dataset.pt", weights_only=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Ground Truth (Original unmasked values)
    sim_data = np.load(data_dir / "uav_sim_interim.npz")
    # 'snr_scaled' contains the ground truth values [T, N]
    snr_truth = sim_data["snr_scaled"] 
    # Slice to match the test window
    snr_truth_subset = snr_truth[args.start_idx : args.end_idx]
    
    # 2. Load Model
    num_features = dataset[0][0].x.shape[1]
    num_uavs = dataset[0][2].shape[0]
    
    model = Autoencoder(train_features=num_features, 
                        latent_dim=args.latent_dim, 
                        num_uavs=num_uavs).to(device)
    
    model_path = data_dir / "recon_model.pt"
    if not model_path.exists():
        raise FileNotFoundError("Run train_recon.py first.")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Evaluation Loop
    mse_actual_list = []
    mse_recon_list = []
    
    values_pred = []
    values_gt = []

    print("Evaluating...")
    
    # We iterate through the loader. 
    # Note: len(loader) should match len(snr_truth_subset)
    data_list = list(loader)
    
    for i, (data, target_latent, target_recon) in enumerate(data_list):
        if i >= len(snr_truth_subset): break

        data = data.to(device)
        target_recon = target_recon.to(device)

        with torch.no_grad():
            latent, recon = model(data.x, data.edge_index, data.batch)

        # Handle shapes
        if recon.size() != target_recon.size():
             if target_recon.size(0) == recon.size(1):
                 target_recon = target_recon.transpose(0, 1)
             else:
                 target_recon = target_recon.view_as(recon)

        # Get Ground Truth for this step
        # Shape [N] or [N, 1]
        actual_nodes = torch.tensor(snr_truth_subset[i], dtype=torch.float32).to(device)
        if actual_nodes.ndim == 1:
            actual_nodes = actual_nodes.view(-1, 1)

        # Calculate MSE
        # Note: Your snippet calculated MSE on 'indices' (masked nodes). 
        # Since we don't have the indices saved, we calculate on ALL nodes here.
        # This provides a global reconstruction error.
        
        # 1. MSE vs Ground Truth (Actual unmasked nodes)
        loss_actual = F.mse_loss(recon, actual_nodes)
        mse_actual_list.append(loss_actual.item())

        # 2. MSE vs Koopman Prediction (Recon List)
        loss_recon = F.mse_loss(recon, target_recon)
        mse_recon_list.append(loss_recon.item())

        values_pred.append(recon.cpu().numpy().flatten())
        values_gt.append(actual_nodes.cpu().numpy().flatten())

    # 4. Summary
    run_mse_actual = np.mean(mse_actual_list)
    run_mse_recon  = np.mean(mse_recon_list)

    print(f"Final Results over {len(mse_actual_list)} steps:")
    print(f"MSE vs Actual Nodes (Ground Truth): {run_mse_actual:.8f}")
    print(f"MSE vs Koopman Preds (Input):       {run_mse_recon:.8f}")

    # Save results
    np.savez(data_dir / "recon_eval_results.npz", 
             pred=np.array(values_pred), 
             truth=np.array(values_gt),
             mse_actual=run_mse_actual)
    print(f"Saved evaluation arrays to {data_dir / 'recon_eval_results.npz'}")

if __name__ == "__main__":
    main()