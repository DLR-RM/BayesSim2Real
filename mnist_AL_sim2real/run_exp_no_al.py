import argparse
import functools
import itertools
import torch
import os
import torch.utils.data as data

from src.utils import laaos
import src.utils.torch_utils as torch_utils
from src.datasets.ds_functions import DatasetEnum, get_experiment_data, get_targets
from src.utils.structure import RandomFixedLengthSampler, ActiveLearningData
# from utils.train_model import train_model

import prettyprinter as pp
import logging
import sys

def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description="Pure training loop without AL",
        formatter_class=functools.partial(argparse.ArgumentDefaultsHelpFormatter, width=120),
    )
    parser.add_argument("--batch_size", type=int, default=64, help="input batch size for training")
    parser.add_argument("--test_batch_size", type=int, default=256, help="input batch size for testing")
    parser.add_argument("--validation_set_size", type=int, default=128, help="validation set size")
    parser.add_argument(
        "--early_stopping_patience", type=int, default=1, help="# patience epochs for early stopping per iteration"
    )
    parser.add_argument("--epochs", type=int, default=30, help="number of epochs to train")
    parser.add_argument("--epoch_samples", type=int, default=5056, help="number of epochs to train")
    parser.add_argument(
        "--balanced_validation_set",
        action="store_true",
        default=False,
        help="uses a balanced validation set (instead of randomly picked)"
        "(and if no validation set is provided by the dataset)",
    )
    parser.add_argument("--num_inference_samples", type=int, default=5, help="number of samples for inference")
    parser.add_argument("--no_cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument(
        "--name", type=str, default="results", help="name for the results file (name of the experiment)"
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--train_dataset_limit",
        type=int,
        default=0,
        help="how much of the training set to use for training after splitting off the validation set (0 for all)",
    )
    parser.add_argument(
        "--balanced_training_set",
        action="store_true",
        default=False,
        help="uses a balanced training set (instead of randomly picked)"
        "(and if no validation set is provided by the dataset)",
    )
    parser.add_argument(
        "--balanced_test_set",
        action="store_true",
        default=False,
        help="force balances the test set---use with CAUTION!",
    )
    parser.add_argument(
        "--log_interval", type=int, default=10, help="how many batches to wait before logging training status"
    )
    parser.add_argument(
        "--exp-root",
        type=str,
        help="folder to save training logs.",
    )
    parser.add_argument(
        "--dataset",
        type=DatasetEnum,
        default=DatasetEnum.mnist,
        help=f"dataset to use (options: {[f.name for f in DatasetEnum]})",
    )
    args = parser.parse_args()

    Exp_root = args.exp_root
    os.makedirs(Exp_root, exist_ok=True)
    Exp_log_folder = os.path.join(Exp_root, "init_training_log", f"{args.dataset.name}/")
    os.makedirs(Exp_log_folder, exist_ok=True)
    store = laaos.open_file_store(
        args.name,
        prefix=Exp_log_folder,
        suffix="",
        truncate=False,
        type_handlers=(laaos.StrEnumHandler(), laaos.ToReprHandler()),
    )
    store["args"] = args.__dict__

    print(args.__dict__)

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    print(f"Using {device} for computations")

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    dataset: DatasetEnum = args.dataset

    data_source = dataset.get_data_source()

    reduced_train_length = args.train_dataset_limit

    experiment_data = get_experiment_data(
        data_source,
        dataset.num_classes,
        None,
        False,
        0,
        args.validation_set_size,
        args.balanced_test_set,
        args.balanced_validation_set,
    )

    if not reduced_train_length:
        reduced_train_length = len(experiment_data.available_dataset)

    print(f"Training with reduced dataset of {reduced_train_length} data points")
    if not args.balanced_training_set:
        experiment_data.active_learning_data.acquire(
            experiment_data.active_learning_data.get_random_available_indices(reduced_train_length)
        )
    else:
        print("Using a balanced training set.")
        num_samples_per_class = reduced_train_length // dataset.num_classes
        experiment_data.active_learning_data.acquire(
            list(
                itertools.chain.from_iterable(
                    torch_utils.get_balanced_sample_indices(
                        get_targets(experiment_data.available_dataset), dataset.num_classes, num_samples_per_class
                    ).values()
                )
            )
        )

    if len(experiment_data.train_dataset) < args.epoch_samples:
        sampler = RandomFixedLengthSampler(experiment_data.train_dataset, args.epoch_samples)
    else:
        sampler = data.RandomSampler(experiment_data.train_dataset)

    test_loader = torch.utils.data.DataLoader(
        experiment_data.test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs
    )
    train_loader = torch.utils.data.DataLoader(
        experiment_data.train_dataset, sampler=sampler, batch_size=args.batch_size, **kwargs
    )

    validation_loader = torch.utils.data.DataLoader(
        experiment_data.validation_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs
    )

    def desc(name):
        return lambda engine: "%s" % name

    Exp_model_save_folder = os.path.join(Exp_root, "pretrained_models", f"{args.dataset.name}")
    os.makedirs(Exp_model_save_folder, exist_ok=True)
    if os.path.exists(os.path.join(Exp_model_save_folder, 'model_final.pth')):
        load_model_from_pth = os.path.join(Exp_model_save_folder, 'model_final.pth')
    else:
        load_model_from_pth = None 
        
    dataset.train_model(
        train_loader=train_loader,
        test_loader=test_loader,
        validation_loader=validation_loader,
        num_inference_samples=args.num_inference_samples,
        max_epochs=args.epochs,
        lr=1e-3, # 1e-3, 1e-4
        early_stopping_patience=args.early_stopping_patience,
        desc=desc,
        log_interval=args.log_interval,
        device=device,
        epoch_results_store=store,
        model_save_folder=Exp_model_save_folder,
        load_model_from_pth=load_model_from_pth,
    )


if __name__ == "__main__":
    main()
