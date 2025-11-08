import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool

class GraphSAGERegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv(3, 32)
        self.conv2 = SAGEConv(32, 32)
        self.readout = nn.Linear(32, 1)

    def forward(self, data):
        x = torch.relu(self.conv1(data.x, data.edge_index))
        x = torch.relu(self.conv2(x, data.edge_index))
        # single-graph case: batch is all zeros
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        hg = global_mean_pool(x, batch)
        return self.readout(hg).squeeze(-1)

class GCNRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 32)
        self.conv2 = GCNConv(32, 32)
        self.readout = nn.Linear(32, 1)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = torch.relu(self.conv1(x, edge_index, edge_weight))
        x = torch.relu(self.conv2(x, edge_index, edge_weight))
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        hg = global_mean_pool(x, batch)
        return self.readout(hg).squeeze(-1)