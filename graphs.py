from pathlib import Path
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data

def graph_to_data(graph_path: Path, age: float, edge_key="number_of_fibers") -> Data:
    G = nx.read_graphml(graph_path)
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}

    # node features: x, y, z coordinates
    coords = np.zeros((len(nodes), 3), dtype=np.float32)
    for n in nodes:
        a = G.nodes[n]
        coords[idx[n]] = [
            float(a["dn_position_x"]),
            float(a["dn_position_y"]),
            float(a["dn_position_z"]),
        ]
    # replace NaNs with 0.0 (some graphs have NaNs) (can be updated to other methods)
    coords = np.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)
    x = torch.tensor(coords, dtype=torch.float32)

    # edges: undirected + raw edge weights
    src, dst, ew = [], [], []
    edge_attrs_list = []

    edge_feature_keys = None

    for u, v, d in G.edges(data=True):
        ui, vi = idx[u], idx[v]

        if edge_feature_keys is None:
            numeric_keys = []
            for k, val in d.items():
                try:
                    float(val)
                    numeric_keys.append(k)
                except (TypeError, ValueError):
                    continue
            edge_feature_keys = numeric_keys

        if edge_key in d:
            w = float(d[edge_key])
        else:
            w = 0.0

        src.extend([ui, vi])
        dst.extend([vi, ui])
        ew.extend([w, w])

        if edge_feature_keys:
            feat_vec = [float(d.get(k, np.nan)) for k in edge_feature_keys]
            edge_attrs_list.append(feat_vec)
            edge_attrs_list.append(feat_vec)

    edge_index = torch.tensor([src, dst], dtype=torch.long)

    edge_weight = torch.tensor(
        np.nan_to_num(np.array(ew, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0),
        dtype=torch.float32,
    )
    if edge_attrs_list:
        edge_attr_arr = np.array(edge_attrs_list, dtype=np.float32)
        edge_attr_arr = np.nan_to_num(edge_attr_arr, nan=0.0, posinf=0.0, neginf=0.0)
        edge_attr = torch.tensor(edge_attr_arr, dtype=torch.float32)
    else:
        edge_attr = None

    # label: age
    y = torch.tensor([float(age)], dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, y=y, edge_weight=edge_weight, edge_attr=edge_attr)
    return data