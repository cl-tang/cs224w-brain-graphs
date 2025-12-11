"""GNN backbone configuration."""

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class BackboneConfig:
    """GNN architecture config. Block: Conv -> Norm -> Act -> Dropout -> (+Residual)"""

    # Conv type
    conv_type: Literal["gcn", "sage", "gin", "gine", "gat", "nnconv", "transformer"] = "gcn"
    num_layers: int = 3
    hidden_dim: int = 64
    num_heads: int = 4  # for gat/transformer

    # Block settings
    dropout: float = 0.2
    norm_type: Literal["batch", "layer", "graph", "none"] = "batch"
    skip_type: Literal["none", "residual"] = "residual"
    activation: Literal["relu", "gelu", "elu", "leaky_relu", "silu"] = "relu"

    # Readout
    readout_type: Literal["mean", "sum", "max", "attention"] = "mean"

    # Edge features
    use_edge_weight: bool = True
    use_edge_attr: bool = False

    # Prediction head
    head_hidden_dim: Optional[int] = None
    head_num_layers: int = 0
    head_activation: Literal["relu", "gelu", "elu", "leaky_relu", "silu", "none"] = "relu"
    head_dropout: float = 0.1

    def __post_init__(self):
        # GCN/SAGE/GIN don't use edge_attr
        if self.use_edge_attr and self.conv_type in ("gcn", "sage", "gin"):
            object.__setattr__(self, 'use_edge_attr', False)

        # Validate attention head dimensions
        if self.conv_type in ("gat", "transformer"):
            if self.hidden_dim % self.num_heads != 0:
                raise ValueError(f"hidden_dim must be divisible by num_heads for {self.conv_type}")

        if self.head_num_layers > 0 and self.head_hidden_dim is None:
            object.__setattr__(self, 'head_hidden_dim', self.hidden_dim)

    def __str__(self):
        extra = f", heads={self.num_heads}" if self.conv_type in ("gat", "transformer") else ""
        return f"BackboneConfig({self.conv_type}, L={self.num_layers}, H={self.hidden_dim}{extra})"
