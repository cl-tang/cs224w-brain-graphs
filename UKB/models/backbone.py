"""GNN Backbone: stack of GNN blocks."""

import torch.nn as nn
from .blocks import GNNBlock, create_conv_layer
from ..configs import BackboneConfig


class GNNBackbone(nn.Module):
    """Stack of GNN blocks. Conv-agnostic (works with GCN, SAGE, GIN, GAT, etc.)."""

    def __init__(self, config: BackboneConfig, in_channels: int, edge_dim: int = 0):
        super().__init__()
        self.config = config
        self.in_channels = in_channels
        self.edge_dim = edge_dim

        # Project input to hidden_dim if needed
        self.input_proj = nn.Linear(in_channels, config.hidden_dim) \
            if in_channels != config.hidden_dim else None

        # Build blocks
        self.blocks = nn.ModuleList()
        for i in range(config.num_layers):
            conv = create_conv_layer(
                conv_type=config.conv_type,
                in_channels=config.hidden_dim,
                out_channels=config.hidden_dim,
                edge_dim=edge_dim if config.use_edge_attr else 0,
                num_heads=config.num_heads,
            )
            block = GNNBlock(
                conv=conv,
                hidden_dim=config.hidden_dim,
                conv_type=config.conv_type,
                norm_type=config.norm_type,
                dropout=config.dropout,
                residual=(config.skip_type == "residual"),
                activation=config.activation,
            )
            self.blocks.append(block)

    def forward(self, x, edge_index, edge_weight=None, edge_attr=None, batch=None):
        if self.input_proj is not None:
            x = self.input_proj(x)

        for block in self.blocks:
            x = block(
                x, edge_index,
                edge_weight=edge_weight if self.config.use_edge_weight else None,
                edge_attr=edge_attr if self.config.use_edge_attr else None,
                batch=batch,
            )
        return x

    @property
    def out_channels(self):
        return self.config.hidden_dim
