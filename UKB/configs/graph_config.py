"""Graph configuration - identifies which cached dataset to use."""

from dataclasses import dataclass, field
from typing import List, Optional

from .defaults import (
    DEFAULT_EDGE_WEIGHT_KEY, DEFAULT_TOP_K, DEFAULT_LOG_TRANSFORM, AVAILABLE_EDGE_WEIGHTS,
)


@dataclass
class GraphConfig:
    """Dataset cache identifier: parcellation + edge weight + sparsification."""

    parcellation: str
    edge_weight_key: str = DEFAULT_EDGE_WEIGHT_KEY
    edge_attr_keys: Optional[List[str]] = field(default_factory=lambda: AVAILABLE_EDGE_WEIGHTS.copy())
    top_k: int = DEFAULT_TOP_K
    density: Optional[float] = None
    log_transform: bool = DEFAULT_LOG_TRANSFORM

    def __str__(self):
        sparsif = f"top_k={self.top_k}" if self.top_k > 0 else f"density={self.density}"
        n_attrs = len(self.edge_attr_keys) if self.edge_attr_keys else 0
        return f"GraphConfig({self.parcellation}, {self.edge_weight_key}, {sparsif}, ea={n_attrs})"
