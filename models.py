import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool


class GraphSAGERegressor(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_channels: int = 32, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.convs = nn.ModuleList()
        input_dim = in_channels
        for _ in range(num_layers):
            self.convs.append(SAGEConv(input_dim, hidden_channels))
            input_dim = hidden_channels
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = torch.relu(conv(x, edge_index))
            x = self.dropout(x)
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        hg = global_mean_pool(x, batch)
        return self.readout(hg).squeeze(-1)


class GCNRegressor(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_channels: int = 32, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.convs = nn.ModuleList()
        input_dim = in_channels
        for _ in range(num_layers):
            self.convs.append(GCNConv(input_dim, hidden_channels))
            input_dim = hidden_channels
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_weight = getattr(data, "edge_weight", None)
        for conv in self.convs:
            x = torch.relu(conv(x, edge_index, edge_weight=edge_weight))
            x = self.dropout(x)
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        hg = global_mean_pool(x, batch)
        return self.readout(hg).squeeze(-1)
