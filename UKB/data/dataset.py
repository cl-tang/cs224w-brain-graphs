"""PyG InMemoryDataset for brain connectivity graphs."""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, List, Optional, Union

import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

from .graphs import npz_to_pyg_data
from ..configs.defaults import (
    DATA_ROOT, SUBJECTS_FILE, PHENOTYPES_FILE, CACHE_DIR,
    DEFAULT_PARCELLATION, DEFAULT_EDGE_WEIGHT_KEY, DEFAULT_TOP_K,
    DEFAULT_THRESHOLD, DEFAULT_LOG_TRANSFORM, AVAILABLE_EDGE_WEIGHTS,
)


class BrainGraphDataset(InMemoryDataset):
    """
    In-memory dataset for brain connectivity graphs.
    Loads from NPZ files and caches processed graphs to disk.
    """

    def __init__(
        self,
        parcellation=DEFAULT_PARCELLATION,
        subjects_file=SUBJECTS_FILE,
        phenotypes_file=PHENOTYPES_FILE,
        data_root=DATA_ROOT,
        edge_weight_key=DEFAULT_EDGE_WEIGHT_KEY,
        edge_attr_keys=None,
        threshold=DEFAULT_THRESHOLD,
        density=None,
        top_k=DEFAULT_TOP_K,
        log_transform=DEFAULT_LOG_TRANSFORM,
        cache_dir=CACHE_DIR,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload=False,
    ):
        if edge_attr_keys is None:
            edge_attr_keys = AVAILABLE_EDGE_WEIGHTS.copy()

        self.subjects_file = Path(subjects_file)
        self.phenotypes_file = Path(phenotypes_file)
        self.data_root = Path(data_root)
        self.parcellation = parcellation
        self.edge_weight_key = edge_weight_key
        self.edge_attr_keys = edge_attr_keys
        self.threshold = threshold
        self.density = density
        self.top_k = top_k
        self.log_transform = log_transform
        self.cache_dir = Path(cache_dir)
        self._force_reload = force_reload

        self._subject_ids = self._load_subject_ids()
        self._phenotypes = self._load_phenotypes()
        self._valid_subjects = self._get_valid_subjects()

        cache_name = self._get_cache_name()
        root = self.cache_dir / cache_name

        super().__init__(str(root), transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

        print(f"Loaded {len(self)} brain graphs from cache")
        print(f"  Parcellation: {self.parcellation}, Edge weight: {self.edge_weight_key}")
        if self.top_k > 0:
            print(f"  Top-K: {self.top_k}")
        elif self.density:
            print(f"  Density: {self.density}")

    def _load_subject_ids(self):
        if not self.subjects_file.exists():
            raise FileNotFoundError(f"Subjects file not found: {self.subjects_file}")
        with open(self.subjects_file) as f:
            return [line.strip() for line in f if line.strip()]

    def _load_phenotypes(self):
        if not self.phenotypes_file.exists():
            raise FileNotFoundError(f"Phenotypes file not found: {self.phenotypes_file}")
        df = pd.read_csv(self.phenotypes_file, dtype={"eid": str})
        df["eid"] = df["eid"].astype(str)
        if "age_2" in df.columns:
            df = df.rename(columns={"age_2": "age"})
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        return df

    def _get_valid_subjects(self):
        """Get subjects with both NPZ and age data."""
        pheno_idx = self._phenotypes.set_index("eid")
        valid = []
        missing_npz, missing_age = 0, 0

        for sid in self._subject_ids:
            npz_path = self.data_root / f"{sid}_2" / "connectomes" / "matrices" / self.parcellation
            if not npz_path.exists():
                missing_npz += 1
                continue
            if sid not in pheno_idx.index or pd.isna(pheno_idx.loc[sid, "age"]):
                missing_age += 1
                continue
            valid.append((sid, npz_path, float(pheno_idx.loc[sid, "age"])))

        if missing_npz:
            print(f"Warning: {missing_npz} subjects missing NPZ files")
        if missing_age:
            print(f"Warning: {missing_age} subjects missing age data")
        print(f"Found {len(valid)} valid subjects out of {len(self._subject_ids)}")
        return valid

    def _get_cache_name(self):
        parc_name = self.parcellation.replace(".npz", "")
        parts = [parc_name, self.edge_weight_key]

        if self.top_k > 0:
            parts.append(f"topk{self.top_k}")
        elif self.density is not None:
            parts.append(f"dens{self.density}")
        else:
            parts.append(f"thr{self.threshold}")

        if self.log_transform:
            parts.append("log")
        if self.edge_attr_keys:
            parts.append(f"attrs_{'_'.join(sorted(self.edge_attr_keys))}")

        return "_".join(parts)

    @property
    def raw_file_names(self):
        return [f"{sid}_{self.parcellation}" for sid, _, _ in self._valid_subjects]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass  # data expected to exist

    def process(self):
        print(f"Processing {len(self._valid_subjects)} brain graphs...")

        def load_single(args):
            sid, npz_path, age = args
            try:
                data = npz_to_pyg_data(
                    npz_path, age,
                    edge_weight_key=self.edge_weight_key,
                    edge_attr_keys=self.edge_attr_keys,
                    threshold=self.threshold,
                    density=self.density,
                    top_k=self.top_k,
                    log_transform=self.log_transform,
                    subject_id=sid,
                )
                return (sid, data, None)
            except Exception as e:
                return (sid, None, str(e))

        data_list = []
        errors = []
        num_workers = len(os.sched_getaffinity(0)) or 1
        print(f"Using {num_workers} workers")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(load_single, args): args[0] for args in self._valid_subjects}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Loading"):
                sid, data, error = future.result()
                if error:
                    errors.append(f"{sid}: {error}")
                elif data is not None:
                    data_list.append(data)

        if errors:
            print(f"Errors: {len(errors)} subjects")
            for err in errors[:5]:
                print(f"  {err}")

        if not data_list:
            raise ValueError("No valid graphs processed!")

        data_list.sort(key=lambda d: d.subject_id)

        if self.pre_filter:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform:
            data_list = [self.pre_transform(d) for d in data_list]

        self.save(data_list, self.processed_paths[0])
        print(f"Saved {len(data_list)} graphs to cache")

    def get_subject_ids(self):
        return [sid for sid, _, _ in self._valid_subjects]

    def get_ages(self):
        return [age for _, _, age in self._valid_subjects]

    @staticmethod
    def available_edge_weights():
        return AVAILABLE_EDGE_WEIGHTS.copy()
