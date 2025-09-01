import random
class Net(torch.nn.Module):
    def __init__(self, train_features):
        super(Net, self).__init__()
        self.conv1 = SAGEConv(train_features, 16)
        self.conv2 = SAGEConv(16, 20)
        self.dropout = nn.Dropout(p=0.1)
        
        self.activation = torch.nn.LeakyReLU(0.1)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.activation(self.conv2(x, edge_index))
        graph_embedding = global_mean_pool(x, batch)

        return graph_embedding

class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(20,16)  
        self.linear2 = nn.Linear(16, 16)
        self.linear3 = nn.Linear(16, num_uavs)
        self.activation = torch.nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(p=0.1)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.linear3.weight)
    def forward(self, z):
        z = self.activation(self.linear1(z))
        z = self.activation(self.linear2(z))
        z = self.linear3(z)
        z = z.view(-1, num_uavs, 1)  # Adjust output shape as necessary
        return z

class Autoencoder(torch.nn.Module):
    def __init__(self, train_features):
        super(Autoencoder, self).__init__()
        self.encoder = Net(train_features)
        self.decoder = Decoder()
    
    def forward(self, x, edge_index, batch):
        encoded = self.encoder(x, edge_index, batch)
        reconstructed_x = self.decoder(encoded)
        return reconstructed_x
