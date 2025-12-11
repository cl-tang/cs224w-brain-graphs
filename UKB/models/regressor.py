"""Brain age regression model: backbone + readout + prediction head."""

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from typing import Optional, List

from .blocks import get_readout, get_activation


def build_mlp_head(in_dim, hidden_dim, num_layers, activation="relu", dropout=0.1):
    """Build MLP head. If num_layers=0, returns simple linear layer."""
    if num_layers == 0 or hidden_dim is None:
        return nn.Linear(in_dim, 1)

    layers = []
    # First layer
    layers.extend([
        nn.Linear(in_dim, hidden_dim),
        get_activation(activation),
        nn.Dropout(dropout),
    ])
    # Hidden layers
    for _ in range(num_layers - 1):
        layers.extend([
            nn.Linear(hidden_dim, hidden_dim),
            get_activation(activation),
            nn.Dropout(dropout),
        ])
    # Output
    layers.append(nn.Linear(hidden_dim, 1))
    return nn.Sequential(*layers)


class BrainAgeRegressor(nn.Module):
    """
    Full model: GNN backbone -> graph readout -> MLP head -> age prediction.
    """

    def __init__(self, backbone, readout_type="mean", head_hidden_dim=None,
                 head_num_layers=0, head_activation="relu", head_dropout=0.1):
        super().__init__()
        self.backbone = backbone

        backbone_out = backbone.out_channels
        self.readout = get_readout(readout_type, hidden_dim=backbone_out)
        self.head = build_mlp_head(
            in_dim=backbone_out,
            hidden_dim=head_hidden_dim,
            num_layers=head_num_layers,
            activation=head_activation,
            dropout=head_dropout,
        )

    def forward(self, data: Data) -> Tensor:
        x = data.x
        edge_index = data.edge_index
        edge_weight = getattr(data, 'edge_weight', None)
        edge_attr = getattr(data, 'edge_attr', None)
        batch = getattr(data, 'batch', None)

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = self.backbone(x, edge_index, edge_weight, edge_attr, batch)
        x = self.readout(x, batch)
        return self.head(x).squeeze(-1)

    def forward_with_embeddings(self, data: Data):
        """Return (predictions, graph embeddings) for analysis."""
        x = data.x
        edge_index = data.edge_index
        edge_weight = getattr(data, 'edge_weight', None)
        edge_attr = getattr(data, 'edge_attr', None)
        batch = getattr(data, 'batch', None)

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = self.backbone(x, edge_index, edge_weight, edge_attr, batch)
        embeddings = self.readout(x, batch)
        predictions = self.head(embeddings).squeeze(-1)
        return predictions, embeddings

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return f"BrainAgeRegressor(backbone={self.backbone.__class__.__name__}, params={self.num_parameters:,})"
