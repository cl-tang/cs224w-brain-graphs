"""Feature configuration for node and edge features."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FeatureConfig:
    """
    What features to use from the cached dataset.
    node_features: "none" (1), "basic" (2: degree, strength), "all" (5: + BCT metrics)
    """

    node_features: str = "all"
    use_edge_weight: bool = True
    edge_attr_keys: Optional[List[str]] = None

    def __post_init__(self):
        if self.node_features not in ("none", "basic", "all"):
            raise ValueError("node_features must be none/basic/all")

    @classmethod
    def full_microstructure(cls):
        """All node features + microstructure edge attributes."""
        return cls(
            node_features="all",
            use_edge_weight=True,
            edge_attr_keys=[
                "mean_FA", "mean_MD", "mean_AD", "mean_RD",
                "mean_FW", "mean_NODDI_ICVF", "mean_NODDI_ISOVF", "mean_NODDI_OD",
                "mean_MSK",
            ],
        )

    def get_node_dim(self):
        return {"none": 1, "basic": 2, "all": 5}[self.node_features]

    def get_edge_dim(self):
        return len(self.edge_attr_keys) if self.edge_attr_keys else 0

    def __str__(self):
        ea = len(self.edge_attr_keys) if self.edge_attr_keys else 0
        return f"FeatureConfig(node={self.node_features}, ew={self.use_edge_weight}, ea={ea})"
