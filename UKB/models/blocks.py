"""
GNN building blocks.

Block structure: Conv -> Norm -> Activation -> Dropout -> (+Residual)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, SAGEConv, GINConv, GINEConv,
    GATConv, NNConv, TransformerConv,
    BatchNorm, LayerNorm, GraphNorm,
    global_mean_pool, global_add_pool, global_max_pool,
    GlobalAttention,
)


ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "elu": nn.ELU,
    "leaky_relu": lambda: nn.LeakyReLU(0.1),
    "silu": nn.SiLU,
    "none": nn.Identity,
}

def get_activation(name):
    """Return activation module by name."""
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation: {name}")
    return ACTIVATIONS[name]()


def get_norm_layer(norm_type, dim):
    """Return normalization layer."""
    if norm_type == "batch":
        return BatchNorm(dim)
    elif norm_type == "layer":
        return LayerNorm(dim)
    elif norm_type == "graph":
        return GraphNorm(dim)
    return nn.Identity()


def create_conv_layer(conv_type, in_channels, out_channels, edge_dim=0, num_heads=4):
    """Create a GNN conv layer by type."""

    if conv_type == "gcn":
        return GCNConv(in_channels, out_channels)

    if conv_type == "sage":
        return SAGEConv(in_channels, out_channels)

    if conv_type == "gin":
        mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        return GINConv(mlp)

    if conv_type == "gine":
        mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        return GINEConv(mlp, edge_dim=edge_dim if edge_dim > 0 else None)

    if conv_type == "gat":
        head_dim = out_channels // num_heads
        return GATConv(
            in_channels, head_dim, heads=num_heads, concat=True,
            edge_dim=edge_dim if edge_dim > 0 else None,
        )

    if conv_type == "nnconv":
        edge_nn = nn.Linear(edge_dim if edge_dim > 0 else 1, in_channels * out_channels)
        return NNConv(in_channels, out_channels, edge_nn)

    if conv_type == "transformer":
        head_dim = out_channels // num_heads
        return TransformerConv(
            in_channels, head_dim, heads=num_heads, concat=True,
            edge_dim=edge_dim if edge_dim > 0 else None,
        )

    raise ValueError(f"Unknown conv_type: {conv_type}")


class GNNBlock(nn.Module):
    """Single GNN layer with norm, activation, dropout, and optional residual."""

    def __init__(self, conv, hidden_dim, conv_type, norm_type="batch",
                 dropout=0.2, residual=True, activation="relu"):
        super().__init__()
        self.conv = conv
        self.conv_type = conv_type
        self.norm = get_norm_layer(norm_type, hidden_dim)
        self.norm_type = norm_type
        self.dropout = dropout
        self.residual = residual
        self.activation = get_activation(activation)

    def forward(self, x, edge_index, edge_weight=None, edge_attr=None, batch=None):
        x_in = x

        x = self._apply_conv(x, edge_index, edge_weight, edge_attr)

        # Norm (GraphNorm needs batch)
        if self.norm_type == "graph" and batch is not None:
            x = self.norm(x, batch)
        else:
            x = self.norm(x)

        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.residual and x.shape == x_in.shape:
            x = x + x_in

        return x

    def _apply_conv(self, x, edge_index, edge_weight, edge_attr):
        """Route conv call based on type."""
        if self.conv_type == "gcn":
            return self.conv(x, edge_index, edge_weight=edge_weight)
        if self.conv_type in ("sage", "gin"):
            return self.conv(x, edge_index)
        if self.conv_type in ("gine", "gat", "transformer"):
            return self.conv(x, edge_index, edge_attr=edge_attr)
        if self.conv_type == "nnconv":
            if edge_attr is not None:
                return self.conv(x, edge_index, edge_attr)
            ea = edge_weight.unsqueeze(-1) if edge_weight is not None else \
                 torch.ones(edge_index.size(1), 1, device=x.device)
            return self.conv(x, edge_index, ea)
        return self.conv(x, edge_index)


def get_readout(readout_type, hidden_dim=64):
    """Return graph-level pooling function/module."""
    if readout_type == "mean":
        return global_mean_pool
    if readout_type == "sum":
        return global_add_pool
    if readout_type == "max":
        return global_max_pool
    if readout_type == "attention":
        gate_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        return GlobalAttention(gate_nn)
    raise ValueError(f"Unknown readout_type: {readout_type}")
