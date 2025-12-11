"""Single experiment runner."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from ..configs import GraphConfig, FeatureConfig, BackboneConfig, TrainingConfig
from ..data import BrainGraphDataset
from ..data.subset import create_subset_dataset
from ..models import GNNBackbone, BrainAgeRegressor
from ..evaluation import compute_regression_metrics, plot_training_curves, plot_predictions_scatter

from .splits import SplitManager
from .trainer import Trainer, set_seed


def run_experiment(
    graph_config, feature_config, backbone_config, training_config,
    dataset=None, gpu_id=None, verbose=True,
    output_base_dir="output", checkpoint_base_dir="checkpoints",
) -> Dict[str, Any]:
    """Run complete experiment: load data, train, evaluate, save results."""

    set_seed(training_config.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    # Device
    if gpu_id is not None:
        device = f"cuda:{gpu_id}"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Generate experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{backbone_config.conv_type}_L{backbone_config.num_layers}_H{backbone_config.hidden_dim}_{timestamp}"

    if verbose:
        print(f"\n{'='*60}")
        print(f"Experiment: {experiment_id}")
        print(f"{'='*60}")
        print(f"Device: {device}")
        print(f"Backbone: {backbone_config}")

    # Load dataset
    if dataset is None:
        if verbose:
            print("Loading dataset...")
        dataset = BrainGraphDataset(
            parcellation=graph_config.parcellation,
            edge_weight_key=graph_config.edge_weight_key,
            top_k=graph_config.top_k,
            density=graph_config.density,
            log_transform=graph_config.log_transform,
            edge_attr_keys=graph_config.edge_attr_keys,
        )

    # Create splits
    splits = SplitManager()
    train_idx, val_idx = splits.get_train_val_indices(
        dataset, val_ratio=training_config.val_ratio, seed=training_config.seed
    )
    test_idx = splits.get_test_indices(dataset)

    if verbose:
        print(f"Splits: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    # Create subset datasets
    train_data = create_subset_dataset(
        dataset, train_idx,
        node_features=feature_config.node_features,
        edge_attr_keys=feature_config.edge_attr_keys,
        cached_edge_attr_keys=graph_config.edge_attr_keys,
    )
    val_data = create_subset_dataset(
        dataset, val_idx,
        node_features=feature_config.node_features,
        edge_attr_keys=feature_config.edge_attr_keys,
        cached_edge_attr_keys=graph_config.edge_attr_keys,
    )
    test_data = create_subset_dataset(
        dataset, test_idx,
        node_features=feature_config.node_features,
        edge_attr_keys=feature_config.edge_attr_keys,
        cached_edge_attr_keys=graph_config.edge_attr_keys,
    )

    # Data loaders
    loader_kwargs = dict(batch_size=training_config.batch_size, num_workers=4,
                         pin_memory=True, prefetch_factor=2, persistent_workers=True)
    train_loader = DataLoader(train_data, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_data, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_data, shuffle=False, **loader_kwargs)

    # Build model directly
    in_channels = feature_config.get_node_dim()
    edge_dim = feature_config.get_edge_dim() if backbone_config.use_edge_attr else 0

    backbone = GNNBackbone(backbone_config, in_channels, edge_dim)
    model = BrainAgeRegressor(
        backbone=backbone,
        readout_type=backbone_config.readout_type,
        head_hidden_dim=backbone_config.head_hidden_dim,
        head_num_layers=backbone_config.head_num_layers,
        head_activation=backbone_config.head_activation,
        head_dropout=backbone_config.head_dropout,
    )

    if verbose:
        print(f"Model: {model.num_parameters:,} parameters")

    # Train
    checkpoint_dir = Path(checkpoint_base_dir) / experiment_id
    trainer = Trainer(model, train_loader, val_loader, training_config,
                      device=device, checkpoint_dir=checkpoint_dir)

    if verbose:
        print(f"Training for up to {training_config.epochs} epochs...")
    history = trainer.train(verbose=verbose)
    best_val_loss = trainer.best_val_loss

    # Evaluate
    trainer.load_checkpoint(checkpoint_dir / "best_model.pt")

    val_results = trainer.evaluate(val_loader)
    val_metrics = compute_regression_metrics(val_results['targets'], val_results['predictions'])

    test_results = trainer.evaluate(test_loader)
    test_metrics = compute_regression_metrics(test_results['targets'], test_results['predictions'])

    if verbose:
        print(f"\nResults:")
        print(f"  Val MAE:  {val_metrics['mae']:.3f}, r={val_metrics['pearson_r']:.4f}")
        print(f"  Test MAE: {test_metrics['mae']:.3f}, r={test_metrics['pearson_r']:.4f}")

    # Save plots
    output_dir = Path(output_base_dir) / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_training_curves(history, output_dir / "training_curves.png")
    plot_predictions_scatter(
        test_results['targets'], test_results['predictions'],
        output_dir / "predictions_scatter.png",
        title="Test Set: Predicted vs True Age", metrics=test_metrics,
    )

    if verbose:
        print(f"\nExperiment {experiment_id} completed!")
        print(f"Plots saved to: {output_dir}")

    return {
        "experiment_id": experiment_id,
        "best_val_loss": best_val_loss,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "num_parameters": model.num_parameters,
        "history": history,
        "checkpoint_path": str(checkpoint_dir / "best_model.pt"),
        "output_dir": str(output_dir),
        "device": device,
    }
