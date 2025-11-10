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
    for u, v, d in G.edges(data=True):
        ui, vi = idx[u], idx[v]
        w = float(d[edge_key])
        src += [ui, vi]
        dst += [vi, ui]
        ew  += [w,  w]

    edge_index  = torch.tensor([src, dst], dtype=torch.long)
    # replace NaN edge weights with 0.0 (can be updated to other methods)
    edge_weight = torch.tensor(
        np.nan_to_num(np.array(ew, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0),
        dtype=torch.float32,
    )

    # label: age
    y = torch.tensor([float(age)], dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, y=y, edge_weight=edge_weight)
    return data