import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import SAGEConv, global_mean_pool

class Net(torch.nn.Module):
    def __init__(self, train_features, latent_dim):
        super(Net, self).__init__()
        # Increase the output dimension of conv1
        self.conv1 = SAGEConv(train_features, 128)
        self.conv2 = SAGEConv(128, 32)
        self.dropout = nn.Dropout(p=0.1)
        self.lin1 = Linear(32, 32)
        self.lin2 = Linear(32, latent_dim)

        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)
        
        self.activation = torch.nn.LeakyReLU(0.1)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        graph_embedding = global_mean_pool(x, batch)
        graph_embedding = F.relu(self.lin1(graph_embedding))
        graph_embedding = self.activation(self.lin2(graph_embedding))
        return graph_embedding

class Decoder(torch.nn.Module):
    def __init__(self, latent_dim, num_uavs):
        super(Decoder, self).__init__()
        self.num_uavs = num_uavs
        # Adjust to match the increased latent space size
        self.linear1 = nn.Linear(latent_dim, 32)  
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, num_uavs)
        self.activation = torch.nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(p=0.1)
        
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.linear3.weight)

    def forward(self, z):
        z = self.activation(self.linear1(z))
        z = self.activation(self.linear2(z))
        z = self.linear3(z)
        # Reshape to [Batch_Size, num_uavs, 1] if needed, or just [Batch, num_uavs]
        # Your snippet uses .view(num_uavs, 1), implying batch_size=1
        if z.shape[0] == 1:
            z = z.view(self.num_uavs, 1)
        return z

class Autoencoder(torch.nn.Module):
    def __init__(self, train_features, latent_dim, num_uavs):
        super(Autoencoder, self).__init__()
        self.encoder = Net(train_features, latent_dim)
        self.decoder = Decoder(latent_dim, num_uavs)
    
    def forward(self, x, edge_index, batch):
        encoded = self.encoder(x, edge_index, batch)
        reconstructed_x = self.decoder(encoded)
        return encoded, reconstructed_x