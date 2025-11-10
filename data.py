from pathlib import Path
from typing import List
import warnings

import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset

from graphs import graph_to_data

warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`.*",
    category=FutureWarning,
)


def list_graphs(data_dir: str | Path) -> list[Path]:
    p = Path(data_dir)
    return sorted(p.glob("*.graphml"))

def load_pheno(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"Subject": str, "subject_id": str})
    # rename columns
    if "Subject" in df.columns: df = df.rename(columns={"Subject": "subject_id"})
    if "Age_in_Yrs" in df.columns: df = df.rename(columns={"Age_in_Yrs": "age"})
    df["subject_id"] = df["subject_id"].astype(str)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    return df[["subject_id", "age"]].copy()

def attach_age(file_paths: List[Path], pheno: pd.DataFrame) -> list[float]:
    idx = pheno.set_index("subject_id")["age"]
    if not idx.index.is_unique:
        raise ValueError(f"Duplicate subject_id entries found in phenotype data")
    missing = []
    ages: list[float] = []
    for p in file_paths:
        sid = p.name.split("_")[0]
        if sid not in idx or pd.isna(idx.loc[sid]):
            missing.append(p.name)
        else:
            ages.append(float(idx.loc[sid]))
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"No age found for graph file(s): {joined}")
    if len(ages) != len(file_paths):
        raise ValueError("Age vector length does not match number of graph files.")
    return ages


class BrainGraphDataset(InMemoryDataset):
    """In-memory dataset that caches PyG Data objects constructed from GraphML files."""

    def __init__(
        self,
        graph_dir: str | Path,
        pheno_csv: str | Path,
        edge_key: str = "number_of_fibers",
        cache_dir: str | Path = "cache",
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.graph_dir = Path(graph_dir)
        self.pheno_csv = Path(pheno_csv)
        self.edge_key = edge_key
        self.cache_dir = Path(cache_dir)
        self._graph_files = list_graphs(self.graph_dir)
        if not self._graph_files:
            raise ValueError(f"No .graphml files found in {self.graph_dir}")

        cache_name = f"{self.graph_dir.name}_{self.edge_key}"
        root = self.cache_dir / cache_name
        super().__init__(root=root, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        print(f"Loaded graphs from cache based on {self.graph_dir.name} and {self.edge_key}")

    @property
    def raw_file_names(self) -> List[str]:
        return [p.name for p in self._graph_files]

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt"]

    @property
    def raw_dir(self) -> str:
        return str(self.graph_dir)

    def process(self):
        print(f"Processing graphs from: {self.graph_dir}")
        pheno = load_pheno(self.pheno_csv)
        ages = attach_age(self._graph_files, pheno)

        data_list = [
            graph_to_data(path, age, edge_key=self.edge_key) for path, age in zip(self._graph_files, ages)
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
