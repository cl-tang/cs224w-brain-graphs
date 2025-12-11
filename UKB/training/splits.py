"""Train/val/test split management. Test set is FIXED from file."""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.data import InMemoryDataset

from ..configs.defaults import SUBJECTS_FILE, TEST_SUBJECTS_FILE


class SplitManager:
    """
    Manages subject splits. Test set is fixed and loaded from file.
    Train/val are split from remaining subjects.
    """

    def __init__(self, all_subjects_file=None, test_subjects_file=None):
        if all_subjects_file is None:
            all_subjects_file = str(SUBJECTS_FILE)
        if test_subjects_file is None:
            test_subjects_file = str(TEST_SUBJECTS_FILE)

        self.all_subjects_file = Path(all_subjects_file)
        self.test_subjects_file = Path(test_subjects_file)

        self.all_subjects = self._load_subjects(all_subjects_file)
        self.test_subjects = set(self._load_subjects(test_subjects_file))
        self.trainval_subjects = [s for s in self.all_subjects if s not in self.test_subjects]

        print(f"SplitManager: {len(self.all_subjects)} total, "
              f"{len(self.test_subjects)} test, {len(self.trainval_subjects)} train+val")

    @staticmethod
    def _load_subjects(path):
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]

    def _get_subject_to_idx_map(self, dataset):
        """Build subject_id -> dataset index mapping."""
        return {dataset[i].subject_id: i for i in range(len(dataset))
                if hasattr(dataset[i], 'subject_id')}

    def get_train_val_indices(self, dataset, val_ratio=0.1, seed=42):
        """Get train/val indices, excluding test subjects."""
        subject_to_idx = self._get_subject_to_idx_map(dataset)

        trainval_indices = [subject_to_idx[s] for s in self.trainval_subjects
                           if s in subject_to_idx]

        if not trainval_indices:
            raise ValueError("No train/val subjects found in dataset!")

        train_idx, val_idx = train_test_split(
            trainval_indices, test_size=val_ratio, random_state=seed
        )
        return list(train_idx), list(val_idx)

    def get_test_indices(self, dataset):
        """Get test indices (fixed test set)."""
        subject_to_idx = self._get_subject_to_idx_map(dataset)
        return [subject_to_idx[s] for s in self.test_subjects if s in subject_to_idx]
