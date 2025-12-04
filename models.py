import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool, MessagePassing, TransformerConv
from torch_geometric.typing import Adj, OptTensor
from torch import Tensor


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

class EdgeSAGEConv(MessagePassing):
    def __init__(
        self,
        in_channels_x: int,
        in_channels_e: int,
        out_channels: int,
        aggr: str = "mean",
    ):
        super().__init__(aggr=aggr)
        self.lin_neigh = nn.Linear(
            in_channels_x + in_channels_e, out_channels, bias=True
        )
        self.lin_root = nn.Linear(in_channels_x, out_channels, bias=False)
        self.act = nn.ReLU()

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_attr: OptTensor = None,
    ) -> Tensor:
        if edge_attr is None:
            edge_attr = x.new_zeros(edge_index.size(1), 0)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        if edge_attr.numel() == 0:
            m = x_j
        else:
            m = torch.cat([x_j, edge_attr], dim=-1)
        m = self.lin_neigh(m)
        m = self.act(m)
        return m

    def update(self, aggr_out: Tensor, x: Tensor) -> Tensor:
        out = self.lin_root(x) + aggr_out
        return out


class EdgeSAGERegressor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        edge_in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        out_channels: int = 1,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_dim_x = in_channels
        in_dim_e = edge_in_channels

        for _ in range(num_layers):
            self.convs.append(EdgeSAGEConv(in_dim_x, in_dim_e, hidden_channels))
            self.norms.append(nn.BatchNorm1d(hidden_channels))
            in_dim_x = hidden_channels

        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, "edge_attr", None)

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = torch.relu(x)
            x = self.dropout(x)

        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        hg = global_mean_pool(x, batch)
        out = self.readout(hg).squeeze(-1)
        return out
    
class GraphTransformerRegressor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        edge_in_channels: int = 0,
        hidden_channels: int = 64,
        heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
        out_channels: int = 1,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        input_dim = in_channels
        self.edge_dim = edge_in_channels if edge_in_channels is not None else 0

        for _ in range(num_layers):
            conv = TransformerConv(
                in_channels=input_dim,
                out_channels=hidden_channels // heads,
                heads=heads,
                edge_dim=self.edge_dim if self.edge_dim > 0 else None,
                dropout=dropout,
                beta=False,
            )
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_channels))
            input_dim = hidden_channels

        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, "edge_attr", None)

        if self.edge_dim == 0:
            edge_attr_for_conv = None
        else:
            edge_attr_for_conv = edge_attr

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr=edge_attr_for_conv)
            x = norm(x)
            x = torch.relu(x)
            x = self.dropout(x)

        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        hg = global_mean_pool(x, batch)
        out = self.readout(hg).squeeze(-1)
        return out