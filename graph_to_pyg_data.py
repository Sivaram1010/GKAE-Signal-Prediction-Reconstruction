from torch_geometric.data import Data
import torch

def graph_to_pyg_data(graph):
    node_features = torch.tensor([graph.nodes[node_id]['sinr'] for node_id in graph.nodes()],
                                  dtype=torch.float).view(-1, 1)  # SINR as a single feature per node
    
    # Edge indices
    edge_index = torch.tensor(list(graph.edges()), dtype=torch.long).t().contiguous()
    
    # Edge weights
    edge_weights = torch.tensor([graph[edge[0]][edge[1]]['weight'] for edge in graph.edges()],
                                 dtype=torch.float)
    
    # Create PyG Data object
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_weights)


