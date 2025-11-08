from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from data import list_graphs, load_pheno, attach_age
from graphs import graph_to_data
from models import GCNRegressor
from utils import device

def run_epoch(model, data_list, optimizer=None):
    loss_fn = nn.L1Loss()
    dev = next(model.parameters()).device
    total = 0.0
    for d in data_list:
        d = d.to(dev)
        pred = model(d)
        loss = loss_fn(pred, d.y)
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total += float(loss.item())
    return total / max(1, len(data_list))

def main():
    dev = device()
    graph_dir = Path("data/HCP/86_nodes")
    pheno_csv = Path("data/HCP/HCP_phenotypes.csv")

    files = list_graphs(graph_dir)
    pheno = load_pheno(pheno_csv)
    ages  = attach_age(files, pheno)

    idx = np.arange(len(files))
    tr, te = train_test_split(idx, test_size=0.2, random_state=0)
    tr, va = train_test_split(tr,  test_size=0.2, random_state=0)

    train_files = [files[i] for i in tr]
    val_files = [files[i] for i in va]
    test_files = [files[i] for i in te]

    train_ages = [ages[i] for i in tr]
    val_ages = [ages[i] for i in va]
    test_ages = [ages[i] for i in te]

    train_data = [graph_to_data(p, a, edge_key="number_of_fibers") for p,a in zip(train_files, train_ages)]
    val_data = [graph_to_data(p, a, edge_key="number_of_fibers") for p,a in zip(val_files, val_ages)]
    test_data = [graph_to_data(p, a, edge_key="number_of_fibers") for p,a in zip(test_files, test_ages)]

    model = GCNRegressor().to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 20):
        tr_mae = run_epoch(model, train_data, optimizer=optimizer)
        va_mae = run_epoch(model, val_data)
        if epoch % 10 == 0:
            print(f"epoch {epoch:3d} | train MAE {tr_mae:.3f} | val MAE {va_mae:.3f}")

    te_mae = run_epoch(model, test_data)
    print(f"\nTest MAE: {te_mae:.3f} years")

if __name__ == "__main__":
    main()
