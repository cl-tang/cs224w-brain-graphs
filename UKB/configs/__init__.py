"""Configuration dataclasses for experiments."""

from .defaults import AVAILABLE_EDGE_WEIGHTS
from .graph_config import GraphConfig
from .feature_config import FeatureConfig
from .backbone_config import BackboneConfig
from .training_config import TrainingConfig

__all__ = [
    "GraphConfig", "FeatureConfig", "BackboneConfig", "TrainingConfig",
    "AVAILABLE_EDGE_WEIGHTS",
]
