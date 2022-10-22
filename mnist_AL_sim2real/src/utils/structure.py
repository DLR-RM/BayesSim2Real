from typing import List
import numpy as np
import torch.utils.data as data
import torch
import dataclasses
import typing

class RandomFixedLengthSampler(data.Sampler):
    """
    Sometimes, you really want to do more with little data without increasing the number of epochs.

    This sampler takes a `dataset` and draws `target_length` samples from it (with repetition).
    """

    def __init__(self, dataset: data.Dataset, target_length):
        super().__init__(dataset)
        self.dataset = dataset
        self.target_length = target_length

    def __iter__(self):
        # Ensure that we don't lose data by accident.
        if self.target_length < len(self.dataset):
            return iter(range(len(self.dataset)))

        return iter((torch.randperm(self.target_length) % len(self.dataset)).tolist())

    def __len__(self):
        return self.target_length

@dataclasses.dataclass
class AcquisitionBatch:
    indices: typing.List[int]
    scores: typing.List[float]
    orignal_scores: typing.Optional[typing.List[float]]


class ActiveLearningData(object):
    """Splits `dataset` into an active dataset and an available dataset."""

    def __init__(self, dataset: data.Dataset):
        super().__init__()
        self.dataset = dataset
        self.active_mask = np.full((len(dataset),), False)
        self.available_mask = np.full((len(dataset),), True)

        self.active_dataset = data.Subset(self.dataset, None)
        self.available_dataset = data.Subset(self.dataset, None)

        self._update_indices()

    def _update_indices(self):
        self.active_dataset.indices = np.nonzero(self.active_mask)[0]
        self.available_dataset.indices = np.nonzero(self.available_mask)[0]

    def get_dataset_indices(self, available_indices: List[int]) -> List[int]:
        indices = self.available_dataset.indices[available_indices]
        return indices

    def acquire(self, available_indices):
        indices = self.get_dataset_indices(available_indices)

        self.active_mask[indices] = True
        self.available_mask[indices] = False
        self._update_indices()

    def make_unavailable(self, available_indices):
        indices = self.get_dataset_indices(available_indices)

        self.available_mask[indices] = False
        self._update_indices()

    def get_random_available_indices(self, size):
        assert 0 <= size <= len(self.available_dataset)
        available_indices = torch.randperm(len(self.available_dataset))[:size]
        return available_indices

    def extract_dataset(self, size):
        """Extract a dataset randomly from the available dataset and make those indices unavailable."""
        return self.extract_dataset_from_indices(self.get_random_available_indices(size))

    def extract_dataset_from_indices(self, available_indices):
        """Extract a dataset from the available dataset and make those indices unavailable."""
        dataset_indices = self.get_dataset_indices(available_indices)

        self.make_unavailable(available_indices)
        return data.Subset(self.dataset, dataset_indices)
