"""Graph loading utilities for brain connectivity data."""

import math
import sys
from pathlib import Path
from typing import List, Optional, Union

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import torch
from torch_geometric.data import Data

from ..configs.defaults import AVAILABLE_EDGE_WEIGHTS

# Optional: BCT (Brain Connectivity Toolbox) for graph metrics
# Install from: https://github.com/aestrivex/bctpy
# Or set BCT_PATH environment variable to your BCT installation
BCT_AVAILABLE = False
try:
    bct_path = os.environ.get("BCT_PATH")
    if bct_path:
        sys.path.append(bct_path)
    from BCT import clustering_coef_wu, eigenvector_centrality_und, modularity_und, participation_coef
    BCT_AVAILABLE = True
except ImportError:
    pass


def reconstruct_symmetric_matrix(array_1d):
    """Reconstruct symmetric matrix from upper triangular 1D array."""
    length = len(array_1d)
    n = int((1 + math.sqrt(1 + 8 * length)) / 2)
    if n * (n - 1) // 2 != length:
        raise ValueError(f"Invalid length {length} for symmetric matrix")

    matrix = np.zeros((n, n), dtype=array_1d.dtype)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i, j] = array_1d[idx]
            matrix[j, i] = array_1d[idx]
            idx += 1
    return matrix


def load_matrices_from_npz(npz_path, edge_keys=None):
    """Load connectivity matrices from NPZ file."""
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    data = np.load(npz_path)
    if edge_keys is None:
        edge_keys = list(data.keys())
    else:
        missing = set(edge_keys) - set(data.keys())
        if missing:
            raise KeyError(f"Keys not found: {missing}")

    return {key: reconstruct_symmetric_matrix(data[key]) for key in edge_keys}


def compute_threshold_for_density(matrix, density):
    """Find threshold that yields target graph density."""
    n = matrix.shape[0]
    max_edges = n * (n - 1) // 2
    upper_tri = np.abs(matrix[np.triu_indices(n, k=1)])
    num_keep = int(density * max_edges)

    if num_keep <= 0:
        return np.inf
    if num_keep >= max_edges:
        return 0.0

    sorted_weights = np.sort(upper_tri)[::-1]
    return sorted_weights[num_keep] if num_keep < len(sorted_weights) else 0.0


def apply_top_k(matrix, top_k):
    """Keep top-K edges per node (symmetric: keep if top-K for either endpoint)."""
    n = matrix.shape[0]
    abs_matrix = np.abs(matrix)
    mask = np.zeros_like(matrix, dtype=bool)

    for i in range(n):
        row = abs_matrix[i, :].copy()
        row[i] = -np.inf  # exclude self-loops
        if top_k >= n - 1:
            top_idx = np.where(row > 0)[0]
        else:
            top_idx = np.argpartition(row, -top_k)[-top_k:]
            top_idx = top_idx[row[top_idx] > 0]
        mask[i, top_idx] = True

    mask = mask | mask.T
    result = matrix.copy()
    result[~mask] = 0.0
    return result


def matrix_to_edge_index(matrix, threshold=0.0):
    """Convert adjacency matrix to edge_index and edge_weight tensors."""
    rows, cols = np.where(np.abs(matrix) > threshold)
    edge_index = torch.from_numpy(np.stack([rows, cols]).astype(np.int64))
    edge_weight = torch.from_numpy(matrix[rows, cols].astype(np.float32))
    return edge_index, edge_weight


def compute_node_features(matrix, threshold=0.0):
    """Compute node features: degree, strength, and optionally BCT metrics."""
    n = matrix.shape[0]
    abs_matrix = np.abs(matrix).astype(np.float64)

    degree = np.sum(np.abs(matrix) > threshold, axis=1).astype(np.float32)
    strength = np.sum(matrix, axis=1).astype(np.float32)

    if BCT_AVAILABLE:
        # BCT features (with fallbacks)
        try:
            clustering = clustering_coef_wu(abs_matrix).astype(np.float32)
        except:
            clustering = np.zeros(n, dtype=np.float32)

        try:
            eigenvec = eigenvector_centrality_und(abs_matrix).flatten().astype(np.float32)
        except:
            eigenvec = np.zeros(n, dtype=np.float32)

        try:
            community, _ = modularity_und(abs_matrix)
            participation = participation_coef(abs_matrix, community).astype(np.float32)
        except:
            participation = np.zeros(n, dtype=np.float32)

        features = np.stack([degree, strength, clustering, eigenvec, participation], axis=1)
    else:
        # Basic features only (degree, strength + 3 zeros for compatibility)
        features = np.stack([
            degree, strength,
            np.zeros(n, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
        ], axis=1)

    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def npz_to_pyg_data(npz_path, age, edge_weight_key="sift2_fbc", edge_attr_keys=None,
                    threshold=0.0, density=None, top_k=0, log_transform=False, subject_id=None):
    """Convert NPZ connectivity file to PyG Data object."""
    npz_path = Path(npz_path)

    # Load matrices
    keys_to_load = [edge_weight_key]
    if edge_attr_keys:
        keys_to_load.extend([k for k in edge_attr_keys if k != edge_weight_key])
    matrices = load_matrices_from_npz(npz_path, keys_to_load)

    primary_matrix = matrices[edge_weight_key].copy()
    num_nodes = primary_matrix.shape[0]

    # Sparsification (top_k takes precedence)
    if top_k > 0:
        primary_matrix = apply_top_k(primary_matrix, top_k)
        effective_threshold = 0.0
    elif density is not None and 0 < density < 1:
        effective_threshold = compute_threshold_for_density(primary_matrix, density)
        primary_matrix[np.abs(primary_matrix) <= effective_threshold] = 0.0
        effective_threshold = 0.0
    else:
        effective_threshold = threshold

    # Log transform
    if log_transform:
        nonzero = primary_matrix != 0
        signs = np.sign(primary_matrix)
        primary_matrix = np.where(nonzero, np.log1p(np.abs(primary_matrix)) * signs, 0.0)

    # Extract edges
    edge_index, edge_weight = matrix_to_edge_index(primary_matrix, effective_threshold)
    edge_weight = torch.nan_to_num(edge_weight, nan=0.0, posinf=0.0, neginf=0.0)

    # Edge attributes
    edge_attr = None
    if edge_attr_keys:
        src, dst = edge_index[0].numpy(), edge_index[1].numpy()
        attr_list = []
        for key in edge_attr_keys:
            attr_matrix = matrices.get(key)
            if attr_matrix is None:
                attr_matrix = load_matrices_from_npz(npz_path, [key])[key]
            attr_list.append(attr_matrix[src, dst])
        edge_attr = np.stack(attr_list, axis=1).astype(np.float32)
        edge_attr = torch.tensor(np.nan_to_num(edge_attr, nan=0.0, posinf=0.0, neginf=0.0))

    # Node features
    thresholded = primary_matrix.copy()
    thresholded[np.abs(thresholded) <= effective_threshold] = 0.0
    x = torch.tensor(compute_node_features(thresholded, effective_threshold), dtype=torch.float32)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=torch.tensor([float(age)], dtype=torch.float32),
        num_nodes=num_nodes,
    )
    if edge_attr is not None:
        data.edge_attr = edge_attr
    if subject_id is not None:
        data.subject_id = subject_id

    return data


def get_parcellation_info(npz_filename):
    """Get parcellation metadata from filename."""
    parcellations = {
        "Glasser+Tian_Subcortex_S1_3T.npz": {"cortical": "Glasser", "subcortical": "Tian_S1", "expected_nodes": 376},
        "Glasser+Tian_Subcortex_S4_3T.npz": {"cortical": "Glasser", "subcortical": "Tian_S4", "expected_nodes": 414},
        "Schaefer7n100p+Tian_Subcortex_S1_3T.npz": {"cortical": "Schaefer_100", "subcortical": "Tian_S1", "expected_nodes": 116},
        "Schaefer7n300p+Tian_Subcortex_S1_3T.npz": {"cortical": "Schaefer_300", "subcortical": "Tian_S1", "expected_nodes": 316},
        "Schaefer7n500p+Tian_Subcortex_S4_3T.npz": {"cortical": "Schaefer_500", "subcortical": "Tian_S4", "expected_nodes": 554},
        "Schaefer7n800p+Tian_Subcortex_S4_3T.npz": {"cortical": "Schaefer_800", "subcortical": "Tian_S4", "expected_nodes": 854},
        "Schaefer7n1000p+Tian_Subcortex_S4_3T.npz": {"cortical": "Schaefer_1000", "subcortical": "Tian_S4", "expected_nodes": 1054},
        "Schaefer17n1000p+Tian_Subcortex_S4_3T.npz": {"cortical": "Schaefer_17n_1000", "subcortical": "Tian_S4", "expected_nodes": 1054},
    }
    return parcellations.get(npz_filename, {"cortical": "unknown", "subcortical": "unknown"})
