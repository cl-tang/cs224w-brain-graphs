"""Default values and paths."""

from pathlib import Path

# Data paths - UPDATE THESE FOR YOUR SETUP
DATA_ROOT = Path("data/raw/subjects")  # Directory containing subject subdirectories
SUBJECTS_FILE = Path("data/raw/subjects.txt")  # List of subject IDs
TEST_SUBJECTS_FILE = Path("data/raw/test_subjects.txt")  # Fixed test set
PHENOTYPES_FILE = Path("data/raw/phenotypes.csv")  # CSV with 'eid' and 'age' columns

CACHE_DIR = Path("cache")
OUTPUT_DIR = Path("output")
CHECKPOINT_DIR = Path("checkpoints")

# Graph defaults
DEFAULT_PARCELLATION = "Glasser+Tian_Subcortex_S4_3T.npz"
DEFAULT_EDGE_WEIGHT_KEY = "sift2_fbc_norm"
DEFAULT_TOP_K = 0
DEFAULT_DENSITY = None
DEFAULT_THRESHOLD = 0.0
DEFAULT_LOG_TRANSFORM = False

# All edge weight metrics (order defines edge_attr columns)
AVAILABLE_EDGE_WEIGHTS = [
    "mean_FA", "mean_MD", "mean_AD", "mean_RD", "mean_FW",
    "mean_NODDI_ICVF", "mean_NODDI_ISOVF", "mean_NODDI_OD", "mean_MSK",
    "streamline_count_norm", "streamline_count", "sift2_fbc", "sift2_fbc_norm",
]

# Training defaults
DEFAULT_SEED = 42
DEFAULT_VAL_RATIO = 0.1
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 100
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_PATIENCE = 15
