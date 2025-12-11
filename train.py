from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from torch.utils.data import Subset

from alters import AlterSRegressor
from data import BrainGraphDataset
from models import GCNRegressor, GraphSAGERegressor, EdgeSAGERegressor, GraphTransformerRegressor
from utils import device
from paths import PHENOTYPES_CSV, GRAPH_DIR
from plot import plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_resolution", type=str, default="86", help="Number of nodes in the graph")
    parser.add_argument("--model", choices=["gcn", "sage", "edgesage", "transformer", "alter-s"], default="gcn")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--edge-key", type=str, default="number_of_fibers")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--cache-dir", type=str, default="cache", help="Directory to store cached PyG tensors")
    return parser.parse_args()


def run_epoch(model, loader, optimizer=None):
    loss_fn = nn.L1Loss()
    dev = next(model.parameters()).device
    total_loss = 0.0
    total_mae = 0.0
    total_graphs = 0
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = batch.to(dev)
            pred = model(batch)
            loss = loss_fn(pred, batch.y)
            mae = torch.mean(torch.abs(pred.detach() - batch.y.detach()))
            batch_graphs = batch.num_graphs
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * batch_graphs
            total_mae += float(mae.item()) * batch_graphs
            total_graphs += batch_graphs
    avg_loss = total_loss / max(1, total_graphs)
    avg_mae = total_mae / max(1, total_graphs)
    return avg_loss, avg_mae

def main():
    args = parse_args()
    dev = device()
    graph_dir = Path(GRAPH_DIR+f"/{args.graph_resolution}_nodes")
    pheno_csv = Path(PHENOTYPES_CSV)

    dataset = BrainGraphDataset(
        graph_dir=graph_dir,
        pheno_csv=pheno_csv,
        edge_key=args.edge_key,
        cache_dir=args.cache_dir,
    )

    edge_attr = getattr(dataset[0], "edge_attr", None)
    if edge_attr is not None:
        edge_in_dim = edge_attr.size(-1)
    else:
        edge_in_dim = 0

    idx = np.arange(len(dataset))
    tr, te = train_test_split(idx, test_size=0.2, random_state=0)
    tr, va = train_test_split(tr, test_size=0.2, random_state=0)

    train_dataset = Subset(dataset, tr.tolist())
    val_dataset = Subset(dataset, va.tolist())
    test_dataset = Subset(dataset, te.tolist())

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    in_dim = dataset.num_node_features
    print(f"Model: {args.model.upper()}Regressor, Hidden Dim: {args.hidden_dim}, Num Layers: {args.num_layers}, Dropout: {args.dropout}")
    if args.model == "gcn":
        model = GCNRegressor(
            in_channels=in_dim,
            hidden_channels=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    elif args.model == "sage":
        model = GraphSAGERegressor(
            in_channels=in_dim,
            hidden_channels=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    elif args.model == "edgesage":
        edge_in_dim = dataset[0].edge_attr.size(-1)
        model = EdgeSAGERegressor(
            in_channels=in_dim,
            edge_in_channels=edge_in_dim,
            hidden_channels=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    elif args.model == "transformer":
        edge_in_dim = dataset[0].edge_attr.size(-1)
        model = GraphTransformerRegressor(
            in_channels=in_dim,
            edge_in_channels=edge_in_dim,
            hidden_channels=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            heads=4,
        )
    elif args.model == "alter-s":
        model = AlterSRegressor(
            in_channels=in_dim,
            edge_in_channels=edge_in_dim,
            hidden_channels=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            heads=4,
            long_range_steps=4,
            long_range_dim=32,
        )

    model = model.to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_mae = run_epoch(model, train_loader, optimizer=optimizer)
        va_loss, va_mae = run_epoch(model, val_loader)

        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        train_maes.append(tr_mae)
        val_maes.append(va_mae)

        if epoch % 10 == 0 or epoch == args.epochs or epoch == 1:
            print(
                f"epoch {epoch:3d} | train loss {tr_loss:.3f} | val loss {va_loss:.3f} "
                f"| train MAE {tr_mae:.3f} | val MAE {va_mae:.3f}"
            )

    te_loss, te_mae = run_epoch(model, test_loader)
    print(f"\nTest loss (L1): {te_loss:.3f}")
    print(f"Test MAE: {te_mae:.3f} years")
    print("Training finished.")

    epochs = np.arange(1, args.epochs + 1)

    os.makedirs(Path(args.output_dir, "figures"), exist_ok=True)
    plot_training_curves(
        epochs=epochs,
        train_losses=train_losses,
        val_losses=val_losses,
        train_maes=train_maes,
        val_maes=val_maes,
        output_path=Path(args.output_dir, "figures", "training_metrics.png"),
        show=True,
    )

if __name__ == "__main__":
    main()
