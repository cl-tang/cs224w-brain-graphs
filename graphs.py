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
    edge_weight = torch.tensor(ew, dtype=torch.float32)

    # label: age
    y = torch.tensor([float(age)], dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, y=y, edge_weight=edge_weight)
    return data