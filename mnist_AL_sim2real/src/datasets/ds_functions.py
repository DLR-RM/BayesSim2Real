import torch
import os
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils import data as data
from torch import optim
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
from torch.utils.data import Subset
import numpy as np
import collections
from dataclasses import dataclass
import itertools
import enum
from typing import List

from src.utils.structure import ActiveLearningData
from src.datasets.mnistm import MNISTM, rgbMNIST
import src.models.mnist_model as mnist_model
# from torch_utils import get_balanced_sample_indices
from src.utils.torch_utils import get_balanced_sample_indices
from src.models.train_model import train_model

dataset_root = os.environ.get('DIGITS_DATA_SET_PATH', "./digits_data")
os.makedirs(dataset_root, exist_ok=True)

# TODO: I fucked this one up. Get rid of this again. (Need the range here to support slicing!)
def SubrangeDataset(dataset, begin, end):
    if end > len(dataset):
        end = len(dataset)
    return Subset(dataset, range(begin, end))

def dataset_subset_split(dataset, indices):
    if isinstance(indices, int):
        indices = [indices]

    datasets = []

    last_index = 0
    for index in indices:
        datasets.append(SubrangeDataset(dataset, last_index, index))
        last_index = index
    datasets.append(SubrangeDataset(dataset, last_index, len(dataset)))

    return datasets

class TransformedDataset(data.Dataset):
    """
    Transforms a dataset.

    Arguments:
        dataset (Dataset): The whole Dataset
        transformer (LambdaType): (idx, sample) -> transformed_sample
    """

    def __init__(self, dataset, *, transformer=None, vision_transformer=None):
        self.dataset = dataset
        assert not transformer or not vision_transformer
        if transformer:
            self.transformer = transformer
        else:
            self.transformer = lambda _, data_label: (vision_transformer(data_label[0]), data_label[1])

    def __getitem__(self, idx):
        return self.transformer(idx, self.dataset[idx])

    def __len__(self):
        return len(self.dataset)


@dataclass
class ExperimentData:
    active_learning_data: ActiveLearningData
    train_dataset: Dataset
    available_dataset: Dataset
    validation_dataset: Dataset
    test_dataset: Dataset
    initial_samples: List[int]

@dataclass
class DataSource:
    train_dataset: Dataset
    validation_dataset: Dataset = None
    test_dataset: Dataset = None
    shared_transform: object = None
    train_transform: object = None
    scoring_transform: object = None


def get_MNIST():
    # num_classes=10, input_size=28
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = rgbMNIST(dataset_root, train=True, download=True, transform=transform)
    test_dataset = MNISTM(dataset_root, train=False, transform=transform)

    return DataSource(train_dataset=train_dataset, test_dataset=test_dataset)


def get_MNISTM():
    # num_classes=10, input_size=28
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = MNISTM(dataset_root, train=True, download=True, transform=transform)
    test_dataset = MNISTM(dataset_root, train=False, transform=transform)

    return DataSource(train_dataset=train_dataset, test_dataset=test_dataset)


class DatasetEnum(enum.Enum):
    mnist = "mnist"
    mnistm = "mnistm"

    def get_data_source(self):
        if self == DatasetEnum.mnist:
            return get_MNIST()
        elif self == DatasetEnum.mnistm:
            return get_MNISTM()
        else:
            raise NotImplementedError(f"Unknown dataset {self}!")

    @property
    def num_classes(self):
        if self in (
                DatasetEnum.mnist,
                DatasetEnum.mnistm
        ):
            return 10
        else:
            raise NotImplementedError(f"Unknown dataset {self}!")

    def create_bayesian_model(self, device):
        num_classes = self.num_classes
        if self in (
                DatasetEnum.mnist,
                DatasetEnum.mnistm,
        ):
            return mnist_model.BayesianNet(num_classes=num_classes).to(device)
        else:
            raise NotImplementedError(f"Unknown dataset {self}!")

    def create_optimizer(self, model, lr=1e-4):
        if self == DatasetEnum.mnist or self == DatasetEnum.mnistm:
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5) # 1e-3
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr) # 1e-4
        return optimizer

    def create_train_model_extra_args(self, optimizer):
        return {}

    def train_model(
            self,
            train_loader,
            test_loader,
            validation_loader,
            num_inference_samples,
            max_epochs,
            early_stopping_patience,
            desc,
            log_interval,
            device,
            lr=1e-3,
            WarmupLRS_flag=False,
            epoch_results_store=None,
            model_save_folder=None,
            load_model_from_pth=None,
    ):
        model = self.create_bayesian_model(device)
        if load_model_from_pth is not None:
            print(f"\n##### Loading trained model weights saved in {load_model_from_pth} #####\n")
            model.load_state_dict(torch.load(load_model_from_pth))
        optimizer = self.create_optimizer(model, lr=lr)
        if WarmupLRS_flag:
            # lr_lambda1 = lambda epoch: epoch/float(max_epochs)
            lr_scheduler = MultiStepLR(optimizer, milestones=[int(max_epochs/3),int(max_epochs/2)], gamma=10)
        else:
            lr_scheduler = None
        num_epochs, test_metrics = train_model(
            model,
            optimizer,
            max_epochs,
            early_stopping_patience,
            num_inference_samples,
            test_loader,
            train_loader,
            validation_loader,
            log_interval,
            desc,
            device,
            lr_scheduler=lr_scheduler,
            epoch_results_store=epoch_results_store,
            model_save_folder=model_save_folder,
            **self.create_train_model_extra_args(optimizer),
        )
        return model, num_epochs, test_metrics

def get_experiment_data(
        data_source,
        num_classes,
        initial_samples,
        reduced_dataset,
        samples_per_class,
        validation_set_size,
        balanced_test_set,
        balanced_validation_set,
):
    train_dataset, test_dataset, validation_dataset = (
        data_source.train_dataset,
        data_source.test_dataset,
        data_source.validation_dataset,
    )

    active_learning_data = ActiveLearningData(train_dataset)
    if initial_samples == -1: # no need of initial samples
        if initial_samples is None:
            initial_samples = list(
                itertools.chain.from_iterable(
                    get_balanced_sample_indices(
                        get_targets(train_dataset), num_classes=num_classes, n_per_digit=samples_per_class
                    ).values()
                )
            )

        # Split off the validation dataset after acquiring the initial samples.
        active_learning_data.acquire(initial_samples)

    if validation_dataset is None:
        print("Acquiring validation set from training set.")
        if not validation_set_size:
            validation_set_size = len(test_dataset)

        if not balanced_validation_set:
            validation_dataset = active_learning_data.extract_dataset(validation_set_size)
        else:
            print("Using a balanced validation set")
            validation_dataset = active_learning_data.extract_dataset_from_indices(
                balance_dataset_by_repeating(
                    active_learning_data.available_dataset, num_classes, validation_set_size, upsample=False
                )
            )
    else:
        if validation_set_size == 0:
            print("Using provided validation set.")
            validation_set_size = len(validation_dataset)
        if validation_set_size < len(validation_dataset):
            print("Shrinking provided validation set.")
            if not balanced_validation_set:
                validation_dataset = data.Subset(
                    validation_dataset, torch.randperm(len(validation_dataset))[:validation_set_size].tolist()
                )
            else:
                print("Using a balanced validation set")
                validation_dataset = data.Subset(
                    validation_dataset,
                    balance_dataset_by_repeating(validation_dataset, num_classes, validation_set_size, upsample=False),
                )

    if balanced_test_set:
        print("Using a balanced test set")
        print("Distribution of original test set classes:")
        classes = get_target_bins(test_dataset)
        print(classes)

        test_dataset = data.Subset(
            test_dataset, balance_dataset_by_repeating(test_dataset, num_classes, len(test_dataset))
        )

    if reduced_dataset:
        # Let's assume we won't use more than 1000 elements for our validation set.
        active_learning_data.extract_dataset(len(train_dataset) - max(len(train_dataset) // 20, 5000))
        test_dataset = SubrangeDataset(test_dataset, 0, max(len(test_dataset) // 10, 5000))
        if validation_dataset:
            validation_dataset = SubrangeDataset(validation_dataset, 0,len(validation_dataset) // 10)
        print("USING REDUCED DATASET!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    show_class_frequencies = True
    if show_class_frequencies:
        print("Distribution of training set classes:")
        classes = get_target_bins(train_dataset)
        print(classes)

        print("Distribution of validation set classes:")
        classes = get_target_bins(validation_dataset)
        print(classes)

        print("Distribution of test set classes:")
        classes = get_target_bins(test_dataset)
        print(classes)

        print("Distribution of pool classes:")
        classes = get_target_bins(active_learning_data.available_dataset)
        print(classes)

        print("Distribution of active set classes:")
        classes = get_target_bins(active_learning_data.active_dataset)
        print(classes)

    print(f"Dataset info:")
    print(f"\t{len(active_learning_data.active_dataset)} active samples")
    print(f"\t{len(active_learning_data.available_dataset)} available samples")
    print(f"\t{len(validation_dataset)} validation samples")
    print(f"\t{len(test_dataset)} test samples")

    if data_source.shared_transform is not None or data_source.train_transform is not None:
        train_dataset = TransformedDataset(
            active_learning_data.active_dataset,
            vision_transformer=compose_transformers([data_source.train_transform, data_source.shared_transform]),
        )
    else:
        train_dataset = active_learning_data.active_dataset

    if data_source.shared_transform is not None or data_source.scoring_transform is not None:
        available_dataset = TransformedDataset(
            active_learning_data.available_dataset,
            vision_transformer=compose_transformers([data_source.scoring_transform, data_source.shared_transform]),
        )
    else:
        available_dataset = active_learning_data.available_dataset

    if data_source.shared_transform is not None:
        test_dataset = TransformedDataset(test_dataset, vision_transformer=data_source.shared_transform)
        validation_dataset = TransformedDataset(validation_dataset, vision_transformer=data_source.shared_transform)

    return ExperimentData(
        active_learning_data=active_learning_data,
        train_dataset=train_dataset,
        available_dataset=available_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        initial_samples=initial_samples,
    )


def compose_transformers(iterable):
    iterable = list(filter(None, iterable))
    if len(iterable) == 0:
        return None
    if len(iterable) == 1:
        return iterable[0]
    return transforms.Compose(iterable)


# TODO: move to utils?
def get_target_bins(dataset):
    classes = collections.Counter(int(target) for target in get_targets(dataset))
    return classes


# TODO: move to utils?
def balance_dataset_by_repeating(dataset, num_classes, target_size, upsample=True):
    balanced_samples_indices = get_balanced_sample_indices(get_targets(dataset), num_classes, len(dataset)).values()

    if upsample:
        num_samples_per_class = max(
            max(len(samples_per_class) for samples_per_class in balanced_samples_indices),
            target_size // num_classes
        )
    else:
        num_samples_per_class = min(
            max(len(samples_per_class) for samples_per_class in balanced_samples_indices),
            target_size // num_classes
        )

    def sample_indices(indices, total_length):
        return (torch.randperm(total_length) % len(indices)).tolist()

    balanced_samples_indices = list(
        itertools.chain.from_iterable(
            [
                [samples_per_class[i] for i in sample_indices(samples_per_class, num_samples_per_class)]
                for samples_per_class in balanced_samples_indices
            ]
        )
    )

    print(
        f"Resampled dataset ({len(dataset)} samples) to a balanced set of {len(balanced_samples_indices)} samples!")

    return balanced_samples_indices


# TODO: move to utils?
def get_targets(dataset):
    """Get the targets of a dataset without any target target transforms(!)."""
    if isinstance(dataset, TransformedDataset):
        return get_targets(dataset.dataset)
    if isinstance(dataset, data.Subset):
        targets = get_targets(dataset.dataset)
        return torch.as_tensor(targets)[dataset.indices]
    if isinstance(dataset, data.ConcatDataset):
        return torch.cat([get_targets(sub_dataset) for sub_dataset in dataset.datasets])

    if isinstance(
            dataset, (datasets.MNIST, MNISTM, datasets.ImageFolder,)
    ):
        return torch.as_tensor(dataset.targets)
    if isinstance(dataset, datasets.SVHN):
        return dataset.labels

    raise NotImplementedError(f"Unknown dataset {dataset}!")
