import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool

NUM_UAVS = 20  # fixed per your setup

class Net(torch.nn.Module):
    def __init__(self, in_features: int = 1):
        super().__init__()
        self.conv1 = SAGEConv(in_features, 16)
        self.conv2 = SAGEConv(16, 20)
        self.activation = torch.nn.LeakyReLU(0.1)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.activation(self.conv2(x, edge_index))
        z = global_mean_pool(x, batch)  # [B, 20]
        return z

class Decoder(torch.nn.Module):
    def __init__(self, latent_dim: int = 20, num_uavs: int = NUM_UAVS):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, 16)
        self.linear2 = nn.Linear(16, 16)
        self.linear3 = nn.Linear(16, num_uavs)
        self.activation = torch.nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(p=0.1)

        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.linear3.weight)

    def forward(self, z):  # z: [B, latent_dim]
        h = self.activation(self.linear1(z))
        h = self.activation(self.linear2(h))
        out = self.linear3(h).unsqueeze(-1)  # [B, num_uavs, 1]
        return out

class Autoencoder(torch.nn.Module):
    def __init__(self, in_features: int = 1, latent_dim: int = 20, num_uavs: int = NUM_UAVS):
        super().__init__()
        self.encoder = Net(in_features)
        self.decoder = Decoder(latent_dim, num_uavs)

    def forward(self, data):
        z = self.encoder(data.x, data.edge_index, data.batch)   # [B, latent_dim]
        x_recon = self.decoder(z)                               # [B, num_uavs, 1]
        return x_recon, z
