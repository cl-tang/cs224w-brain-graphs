# Brain Age Prediction with Graph Neural Networks

A PyTorch Geometric framework for predicting brain age from structural connectivity graphs derived from diffusion MRI tractography.

## Overview

This project implements various GNN architectures (GCN, GraphSAGE, GIN, GAT, Transformer) for brain age prediction using whole-brain structural connectomes.

## Installation

```bash
# Create conda environment
conda create -n pyg python=3.10
conda activate pyg

# Install PyTorch and PyG
pip install torch torchvision
pip install torch-geometric

# Install other dependencies
pip install numpy scipy matplotlib pyyaml
```


## Data Format

### Input Data

Subject connectivity data should be stored as NPZ files with upper-triangular connectivity matrices:

```
data/raw/subjects/
├── subject_001/
│   └── Glasser+Tian_Subcortex_S4_3T.npz
├── subject_002/
│   └── Glasser+Tian_Subcortex_S4_3T.npz
└── ...
```

Each NPZ file should contain edge weight arrays (e.g., `sift2_fbc_norm`, `mean_FA`, `mean_MD`, etc.).

### Required Files

- `data/raw/subjects.txt`: List of subject IDs (one per line)
- `data/raw/phenotypes.csv`: CSV with columns `eid` (subject ID) and `age` (target)

## Usage

### Quick Start

```python
from configs import BackboneConfig, TrainingConfig
from models import GNNBackbone, BrainAgeRegressor
from training.trainer import Trainer

# Configure model
backbone_cfg = BackboneConfig(
    conv_type="gcn",
    hidden_dim=64,
    num_layers=3,
)

# Build model
backbone = GNNBackbone(
    input_dim=5,
    hidden_dim=backbone_cfg.hidden_dim,
    num_layers=backbone_cfg.num_layers,
    conv_type=backbone_cfg.conv_type,
)
model = BrainAgeRegressor(backbone, hidden_dim=backbone_cfg.hidden_dim)

# Train
training_cfg = TrainingConfig(epochs=100, batch_size=32, lr=1e-3)
trainer = Trainer(model, train_loader, val_loader, training_cfg)
trainer.fit()
```

### Running Experiments

```python
from training.experiment import run_experiment
from configs import GraphConfig, FeatureConfig, BackboneConfig, TrainingConfig

results = run_experiment(
    dataset=my_dataset,
    graph_config=GraphConfig(...),
    feature_config=FeatureConfig(),
    backbone_config=BackboneConfig(conv_type="gat", num_layers=4),
    training_config=TrainingConfig(epochs=100),
    output_dir="output/experiment_1"
)
```

## Project Structure

```
├── configs/           # Configuration dataclasses
│   ├── backbone_config.py   # GNN architecture settings
│   ├── feature_config.py    # Node/edge feature selection
│   ├── graph_config.py      # Graph construction settings
│   └── training_config.py   # Training hyperparameters
├── data/              # Data loading and processing
│   ├── dataset.py     # PyG InMemoryDataset
│   ├── graphs.py      # Graph construction from connectomes
│   └── subset.py      # Feature subsetting utilities
├── models/            # Neural network architectures
│   ├── backbone.py    # GNN backbone
│   ├── blocks.py      # GNN layers and components
│   └── regressor.py   # Full regression model
├── training/          # Training infrastructure
│   ├── experiment.py  # Experiment runner
│   ├── splits.py      # Train/val/test splitting
│   └── trainer.py     # Training loop
└── evaluation/        # Evaluation and visualization
    ├── metrics.py     # Regression metrics
    └── plots.py       # Visualization utilities
```

## Supported GNN Architectures

| Type | Description |
|------|-------------|
| `gcn` | Graph Convolutional Network |
| `sage` | GraphSAGE |
| `gin` | Graph Isomorphism Network |
| `gine` | GIN with edge attributes |
| `gat` | Graph Attention Network |
| `nnconv` | Neural Network Convolution |
| `transformer` | Graph Transformer |

## Configuration Options

### BackboneConfig
- `conv_type`: GNN layer type (see table above)
- `hidden_dim`: Hidden dimension (default: 64)
- `num_layers`: Number of GNN layers (default: 3)
- `dropout`: Dropout rate (default: 0.1)
- `skip_type`: Skip connection type (`"none"`, `"residual"`)
- `activation`: Activation function (`"relu"`, `"gelu"`, `"elu"`, etc.)
- `readout`: Graph pooling (`"mean"`, `"sum"`, `"max"`, `"attention"`)

### TrainingConfig
- `epochs`: Maximum training epochs
- `batch_size`: Batch size
- `lr`: Learning rate
- `weight_decay`: L2 regularization
- `patience`: Early stopping patience

## License

MIT

