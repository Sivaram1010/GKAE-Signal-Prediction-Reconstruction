# src/uavnet/recon_utils.py

import numpy as np
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset
import networkx as nx

def generate_masked_indices(num_nodes, num_masked_nodes, previous_indices=None):
    """
    Randomly selects nodes to mask, ensuring no overlap with previously masked nodes.
    """
    if previous_indices is None:
        previous_indices = []
    
    masked_indices = list(previous_indices)

    # Number of new nodes to add
    num_to_add = num_masked_nodes - len(masked_indices)

    if num_to_add > 0:
        candidates = list(set(range(num_nodes)) - set(masked_indices))
        # Safety: don't crash if we ask for more masks than available
        count = min(num_to_add, len(candidates))
        additional_indices = np.random.choice(candidates, count, replace=False)
        masked_indices.extend(additional_indices)

    return masked_indices

def graph_to_pyg_data(graph, masked_indices):
    """
    Converts a NetworkX graph to PyG Data.
    Sets the feature of masked nodes to 0.0.
    """
    # 1. Extract Node Features (SINR)
    # Sort nodes to ensure features align with adjacency matrix order
    node_features_list = [graph.nodes[n]['sinr'] for n in sorted(graph.nodes())]
    node_features = torch.tensor(node_features_list, dtype=torch.float).view(-1, 1)
    
    # 2. Extract Edges and Weights
    edge_indices = []
    edge_weights = []
    
    # Iterate edges
    for u, v in graph.edges():
        edge_indices.append([u, v])
        edge_indices.append([v, u]) # undirected
        
        w = graph[u][v].get('weight', 1.0)
        edge_weights.append(w)
        edge_weights.append(w)
    
    # 3. Apply Masking
    for idx in masked_indices:
        node_features[idx] = 0.0

    # 4. Convert to Tensors
    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0,), dtype=torch.float)

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

class GraphLatentDataset(Dataset):
    """
    Returns: (MaskedGraph, TargetLatent, ReconstructedValues)
    """
    def __init__(self, pyg_data_list, target_latents, reconlist):
        self.pyg_data_list = pyg_data_list
        self.target_latents = target_latents
        self.reconlist = reconlist
        
        assert len(pyg_data_list) == len(target_latents) == len(reconlist), \
            f"Size mismatch: Graph={len(pyg_data_list)}, Z={len(target_latents)}, Recon={len(reconlist)}"

    def __len__(self):
        return len(self.pyg_data_list)

    def __getitem__(self, idx):
        return self.pyg_data_list[idx], self.target_latents[idx], self.reconlist[idx]