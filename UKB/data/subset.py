"""Dataset subsetting utilities for feature selection."""

from typing import List, Optional, Union
import torch
from torch_geometric.data import Data, InMemoryDataset


# Node feature indices: [degree, strength, clustering, eigenvec, participation]
NODE_FEATURE_GROUPS = {
    "none": None,           # constant 1s
    "basic": [0, 1],        # degree, strength
    "bct": [2, 3, 4],       # clustering, eigenvec, participation
    "all": [0, 1, 2, 3, 4],
}


def subset_node_features(data, feature_indices):
    """Subset node features. If "none"/None, use constant 1s."""
    new_data = data.clone()

    if isinstance(feature_indices, str):
        feature_indices = NODE_FEATURE_GROUPS[feature_indices]

    if feature_indices is None:
        new_data.x = torch.ones(data.x.size(0), 1, dtype=data.x.dtype, device=data.x.device)
    else:
        new_data.x = data.x[:, feature_indices].clone()

    return new_data


def subset_edge_attrs(data, attr_indices=None):
    """Subset edge attributes. None=keep all, []=remove all."""
    new_data = data.clone()

    if attr_indices is None:
        return new_data

    if len(attr_indices) == 0:
        if hasattr(new_data, 'edge_attr') and new_data.edge_attr is not None:
            new_data.edge_attr = None
    elif hasattr(data, 'edge_attr') and data.edge_attr is not None:
        new_data.edge_attr = data.edge_attr[:, attr_indices].clone()

    return new_data


def get_edge_attr_indices(cached_keys, selected_keys):
    """Map edge attr key names to indices in cached dataset."""
    indices = []
    for key in selected_keys:
        if key not in cached_keys:
            raise ValueError(f"Key '{key}' not in cached: {cached_keys}")
        indices.append(cached_keys.index(key))
    return indices


def create_subset_dataset(dataset, indices, node_features="all",
                          edge_attr_keys=None, cached_edge_attr_keys=None):
    """Create subset with specified features for train/val/test splits."""
    # Resolve node feature indices
    if isinstance(node_features, str):
        node_feat_idx = NODE_FEATURE_GROUPS[node_features]
    else:
        node_feat_idx = node_features

    # Resolve edge attr indices
    if edge_attr_keys is None or edge_attr_keys is False:
        edge_attr_indices = None
    elif len(edge_attr_keys) == 0:
        edge_attr_indices = []
    elif isinstance(edge_attr_keys[0], str):
        if cached_edge_attr_keys is None:
            raise ValueError("cached_edge_attr_keys required when edge_attr_keys are strings")
        edge_attr_indices = get_edge_attr_indices(cached_edge_attr_keys, edge_attr_keys)
    else:
        edge_attr_indices = edge_attr_keys

    data_list = []
    for i in indices:
        data = dataset[i]
        data = subset_node_features(data, node_feat_idx)
        if edge_attr_indices is not None:
            data = subset_edge_attrs(data, edge_attr_indices)
        data_list.append(data)

    return data_list


def get_feature_dims(dataset, node_features="all", edge_attr_keys=None):
    """Get feature dimensions for model init."""
    sample = dataset[0]

    if isinstance(node_features, str):
        node_feat_idx = NODE_FEATURE_GROUPS[node_features]
    else:
        node_feat_idx = node_features

    node_dim = 1 if node_feat_idx is None else len(node_feat_idx)

    if edge_attr_keys is None:
        edge_dim = sample.edge_attr.shape[1] if hasattr(sample, 'edge_attr') and sample.edge_attr is not None else 0
    elif len(edge_attr_keys) == 0:
        edge_dim = 0
    else:
        edge_dim = len(edge_attr_keys)

    return {"node_dim": node_dim, "edge_dim": edge_dim}
