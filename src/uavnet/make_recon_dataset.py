# src/uavnet/make_recon_dataset.py

import argparse
from pathlib import Path
import numpy as np
import torch
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

from .recon_utils import generate_masked_indices, graph_to_pyg_data, GraphLatentDataset

def rebuild_networkx_graphs(positions, snr_values, radius):
    """
    Rebuilds NetworkX graphs from the saved simulation arrays.
    """
    graphs = []
    print(f"Rebuilding graphs (Radius={radius})...")
    for t in range(len(positions)):
        pos = positions[t] # (N, 2)
        s = snr_values[t]  # (N,)
        
        # Adjacency based on radius
        dist_mat = squareform(pdist(pos))
        adj = dist_mat < radius
        
        # Create Graph
        G = nx.from_numpy_array(adj)
        
        # Add attributes (SINR)
        attrs = {i: {"sinr": float(s[i])} for i in range(len(pos))}
        nx.set_node_attributes(G, attrs)
        graphs.append(G)
    return graphs

def main():
    ap = argparse.ArgumentParser(description="Create Masked Reconstruction Dataset")
    ap.add_argument("--data_dir", type=str, default="data/processed/run1")
    ap.add_argument("--start_idx", type=int, default=300, help="Test set start index")
    ap.add_argument("--end_idx", type=int, default=500, help="Test set end index")
    ap.add_argument("--radius", type=float, default=0.9, help="Must match Step 1 radius")
    ap.add_argument("--mask_ratio", type=float, default=0.4, help="Ratio of nodes to mask")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    
    # 1. Load Simulation Data (to rebuild graphs)
    sim_npz = np.load(data_dir / "uav_sim_interim.npz")
    x_scaled = sim_npz["x_scaled"] # (T, N, 2)
    snr_scaled = sim_npz["snr_scaled"] # (T, N)

    # 2. Rebuild NetworkX Graphs (On the fly)
    # We only need the subset for the test period
    x_subset = x_scaled[args.start_idx : args.end_idx]
    snr_subset = snr_scaled[args.start_idx : args.end_idx]
    
    graphs_subset = rebuild_networkx_graphs(x_subset, snr_subset, args.radius)

    # 3. Load GAE Latents (Ground Truth Z)
    gae_npz = np.load(data_dir / "gae_outputs.npz")
    z = gae_npz["z"]  
    # Scale Z (matches your notebook logic)
    min_val, max_val = np.min(z), np.max(z)
    scaledz = (z - min_val) / (max_val - min_val + 1e-12)
    target_z_subset = scaledz[args.start_idx : args.end_idx]

    # 4. Load Koopman Predictions (Reconstruction Input)
    koop_npz = np.load(data_dir / "koopman_eval_outputs.npz")
    pred_vals = koop_npz["pred"] # (Test_Steps, num_uavs)

    # 5. Align lengths
    n_samples = min(len(graphs_subset), len(target_z_subset), len(pred_vals))
    print(f"Aligning to {n_samples} samples.")
    
    graphs_subset = graphs_subset[:n_samples]
    target_z_subset = target_z_subset[:n_samples]
    pred_vals = pred_vals[:n_samples]

    # 6. Apply Masking
    pyg_data_list = []
    num_nodes = len(graphs_subset[0].nodes())
    num_mask = int(num_nodes * args.mask_ratio)
    print(f"Masking {num_mask} nodes per graph.")

    np.random.seed(42)
    for graph in tqdm(graphs_subset, desc="Masking"):
        masked_indices = generate_masked_indices(num_nodes, num_mask)
        pyg_data = graph_to_pyg_data(graph, masked_indices)
        pyg_data_list.append(pyg_data)

    # 7. Create and Save Dataset
    dataset = GraphLatentDataset(
        pyg_data_list, 
        torch.tensor(target_z_subset, dtype=torch.float32), 
        torch.tensor(pred_vals, dtype=torch.float32)
    )

    save_path = data_dir / "recon_dataset.pt"
    torch.save(dataset, save_path)
    print(f"Saved dataset to {save_path}")

if __name__ == "__main__":
    main()