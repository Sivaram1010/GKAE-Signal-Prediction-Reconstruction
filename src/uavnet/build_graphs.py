import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def build_graphs(coords_TNx2, node_scalar_TN, radius=0.9):
    """
    coords_TNx2: (T, N, 2) positions
    node_scalar_TN: (T, N) per-node scalar feature (e.g., scaled SNR)
    """
    T, N, _ = coords_TNx2.shape
    pyg_list = []
    for t in range(T):
        coords = coords_TNx2[t]
        feats = node_scalar_TN[t]
        G = nx.Graph()
        for i in range(N):
            G.add_node(i, feat=float(feats[i]), xy=coords[i])

        for i in range(N):
            d = np.linalg.norm(coords - coords[i], axis=1)
            d[i] = np.inf
            within = np.where(d <= radius)[0]
            if within.size > 0:
                for j in within:
                    u, v = (i, j) if i < j else (j, i)
                    if not G.has_edge(u, v):
                        G.add_edge(u, v, weight=float(np.linalg.norm(coords[u]-coords[v])))
            else:
                j = int(np.argmin(d))
                u, v = (i, j) if i < j else (j, i)
                if not G.has_edge(u, v):
                    G.add_edge(u, v, weight=float(np.linalg.norm(coords[u]-coords[v])))

        # to PyG
        x = torch.tensor([[G.nodes[i]['feat']] for i in G.nodes()], dtype=torch.float32)
        edges = list(G.edges())
        if len(edges) == 0:
            edge_index = torch.arange(0, len(G), dtype=torch.long).repeat(2,1)
            edge_attr = torch.zeros((len(G),), dtype=torch.float32)
        else:
            ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
            # make undirected explicit
            edge_index = torch.cat([ei, ei.flip(0)], dim=1)
            w = [G[u][v]['weight'] for (u, v) in edges]
            edge_attr = torch.tensor(w + w, dtype=torch.float32)

        pyg_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
    return pyg_list

def make_loaders(pyg_list, train_T=300, val_T=500, batch_size=1, shuffle=True):
    train = DataLoader(pyg_list[:train_T], batch_size=batch_size, shuffle=shuffle)
    val   = DataLoader(pyg_list[:val_T],   batch_size=batch_size, shuffle=False)
    return train, val
