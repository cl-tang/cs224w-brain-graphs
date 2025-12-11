# alters.py

from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import TransformerConv, global_mean_pool


class AlterSRegressor(nn.Module):
    """
    ALTER-S: Structural variant of ALTER.
    - Uses edge_attr to learn adaptive communication strengths f_ij via MLP + sigmoid.
    - Builds a biased random walk kernel R for each graph in the batch.
    - Stacks diag(R^{k-1}) as long-range embeddings E.
    - Projects E -> E_b and concatenates with node features.
    - Feeds [X | E_b] into a TransformerConv stack.
    - Pools and regresses.
    """

    def __init__(
        self,
        in_channels: int,
        edge_in_channels: int,
        hidden_channels: int = 64,
        heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        long_range_steps: int = 4,
        long_range_dim: int = 32,
        out_channels: int = 1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.edge_in_channels = edge_in_channels
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.long_range_steps = long_range_steps
        self.long_range_dim = long_range_dim

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )

        self.long_range_proj = nn.Linear(long_range_steps, long_range_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        input_dim = in_channels + long_range_dim

        for _ in range(num_layers):
            conv = TransformerConv(
                in_channels=input_dim,
                out_channels=hidden_channels // heads,
                heads=heads,
                edge_dim=edge_in_channels,
                dropout=dropout,
                beta=False,
            )
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_channels))
            input_dim = hidden_channels

        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(hidden_channels, out_channels)

    def _compute_long_range_embeddings(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Tensor | None,
    ) -> Tensor:
        """
        Compute E_b for all nodes in a graph
        """
        device = x.device
        dtype = x.dtype
        N = x.size(0)

        if batch is None:
            batch = x.new_zeros(N)
        num_graphs = int(batch.max().item()) + 1

        K = self.long_range_steps
        E = x.new_zeros((N, K), dtype=dtype)
        f_all = torch.sigmoid(self.edge_mlp(edge_attr)).view(-1)

        for g in range(num_graphs):
            node_mask = (batch == g)
            node_idx = node_mask.nonzero(as_tuple=False).view(-1)
            n_g = node_idx.size(0)
            if n_g == 0:
                continue

            global_to_local = -torch.ones(N, dtype, device=device)
            global_to_local[node_idx] = torch.arange(n_g, device=device)
            edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
            if edge_mask.sum() == 0:
                continue

            u_global = edge_index[0, edge_mask]
            v_global = edge_index[1, edge_mask]
            u = global_to_local[u_global]
            v = global_to_local[v_global]
            f = f_all[edge_mask]

            R_g = torch.zeros((n_g, n_g), device=device, dtype=dtype)
            R_g[u, v] += f

            deg = R_g.sum(dim=0, keepdim=True)
            deg[deg == 0] = 1.0
            R_g = R_g / deg

            R_power = torch.eye(n_g, device=device, dtype=dtype)
            for k in range(K):
                diag_vals = torch.diag(R_power)
                E[node_idx, k] = diag_vals
                R_power = R_power @ R_g

        E_b = self.long_range_proj(E)
        return E_b

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, "edge_attr", None)
        batch = getattr(data, "batch", None)

        if edge_attr is None:
            E_b = x.new_zeros(x.size(0), self.long_range_dim)
        else:
            E_b = self._compute_long_range_embeddings(x, edge_index, edge_attr, batch)

        x = torch.cat([x, E_b], dim=-1)

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = torch.relu(x)
            x = self.dropout(x)

        if batch is None:
            batch = x.new_zeros(x.size(0))

        hg = global_mean_pool(x, batch)
        out = self.readout(hg).squeeze(-1)
        return out
