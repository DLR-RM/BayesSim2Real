import argparse
import sys
import torch

from src.utils.acquisition_method import AcquisitionMethod
from src.utils.context_stopwatch import ContextStopwatch
from src.utils.acquisition_functions import AcquisitionFunction
from src.datasets.ds_functions import DatasetEnum, get_targets, get_experiment_data, train_model
from src.utils.structure import RandomFixedLengthSampler
from src.utils.torch_utils import get_base_indices
import src.models.mnist_model as mnist_model
import torch.utils.data as data

from src.utils import laaos
# import blackhc.notebook

import functools
import itertools

import os


def create_experiment_config_argparser(parser):
    parser.add_argument("--exp-root", type=str,  help="folder path to save exp results and logging information.")
    parser.add_argument("--model-path", type=str,  help="path to pre-trained model on simulation data.")
    parser.add_argument("--batch_size", type=int, default=64, help="input batch size for training")
    parser.add_argument("--scoring_batch_size", type=int, default=256, help="input batch size for scoring")
    parser.add_argument("--test_batch_size", type=int, default=256, help="input batch size for testing")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate for active learning")
    parser.add_argument(
        "--validation_set_size",
        type=int,
        default=128,
        help="validation set size (0 for len(test_dataset) or whatever we got from the dataset)",
    )
    parser.add_argument(
        "--early_stopping_patience", type=int, default=1, help="# patience epochs for early stopping per iteration"
    )
    parser.add_argument("--epochs", type=int, default=1, help="number of epochs to train")
    parser.add_argument("--epoch_samples", type=int, default=5056, help="number of epochs to train")
    parser.add_argument("--num_inference_samples", type=int, default=5, help="number of samples for inference")
    parser.add_argument(
        "--available_sample_k",
        type=int,
        default=10,
        help="number of active samples to add per active learning iteration",
    )
    parser.add_argument("--target_num_acquired_samples", type=int, default=800, help="max number of samples to acquire")
    parser.add_argument("--target_accuracy", type=float, default=0.98, help="max accuracy to train to")
    parser.add_argument("--no_cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--quickquick", action="store_true", default=False, help="uses a very reduced dataset")
    parser.add_argument("--seed", nargs="+", default=1, help="random seed")
    parser.add_argument(
        "--log_interval", type=int, default=10, help="how many batches to wait before logging training status"
    )
    parser.add_argument(
        "--initial_samples_per_class",
        type=int,
        default=2,
        help="how many samples per class should be selected for the initial training set",
    )
    parser.add_argument(
        "--initial_sample",
        dest="initial_samples",
        type=int,
        action="append",
        help="sample that needs to be part of the initial samples (instead of sampling initial_samples_per_class)",
        default=None,
    )
    parser.add_argument(
        "--type",
        type=AcquisitionFunction,
        default=AcquisitionFunction.bald,
        help=f"acquisition function to use (options: {[f.name for f in AcquisitionFunction]})",
    )
    parser.add_argument(
        "--acquisition_method",
        type=AcquisitionMethod,
        default=AcquisitionMethod.independent,
        help=f"acquisition method to use (options: {[f.name for f in AcquisitionMethod]})",
    )
    parser.add_argument(
        "--dataset",
        type=DatasetEnum,
        default=DatasetEnum.mnistm,
        help=f"dataset to use (options: {[f.name for f in DatasetEnum]})",
    )
    parser.add_argument(
        "--min_remaining_percentage",
        type=int,
        default=100,
        help="how much of the available dataset should remain after culling in BatchBALD",
    )
    parser.add_argument(
        "--min_candidates_per_acquired_item",
        type=int,
        default=20,
        help="at least min_candidates_per_acquired_item*acqusition_size should remain after culling in BatchBALD",
    )
    parser.add_argument(
        "--initial_percentage",
        type=int,
        default=100,
        help="how much of the available dataset should be kept before scoring (cull randomly for big datasets)",
    )
    parser.add_argument(
        "--reduce_percentage",
        type=int,
        default=0,
        help="how much of the available dataset should be culled after each iteration",
    )
    parser.add_argument(
        "--balanced_validation_set",
        action="store_true",
        default=False,
        help="uses a balanced validation set (instead of randomly picked)"
        "(and if no validation set is provided by the dataset)",
    )
    parser.add_argument(
        "--balanced_test_set",
        action="store_true",
        default=False,
        help="force balances the test set---use with CAUTION!",
    )
    parser.add_argument(
        "--prob_score_sampling",
        action="store_true",
        default=False,
        help="force balances the test set---use with CAUTION!",
    )
    return parser


def main():
    parser = argparse.ArgumentParser(
        description="AL4DA_MNIST", formatter_class=functools.partial(argparse.ArgumentDefaultsHelpFormatter, width=120)
    )
    parser.add_argument("--experiment_task_id", type=str, default=None, help="experiment id")
    parser.add_argument(
        "--experiments_laaos", type=str, default=None, help="Laaos file that contains all experiment task configs"
    )
    parser.add_argument(
        "--experiment_description", type=str, default="Trying stuff..", help="Description of the experiment"
    )
    parser = create_experiment_config_argparser(parser)
    args = parser.parse_args()

    Exp_root = args.exp_root
    Model_saved_root = args.model_path

    for seed_temp in args.seed:
        seed = int(seed_temp)
        torch.manual_seed(seed)

        if args.experiments_laaos is not None:
            config = laaos.safe_load(
                args.experiments_laaos, expose_symbols=(AcquisitionFunction, AcquisitionMethod, DatasetEnum)
            )
            # Merge the experiment config with args.
            # Args take priority.
            args = parser.parse_args(namespace=argparse.Namespace(**config[args.experiment_task_id]))

        # DONT TRUNCATE LOG FILES EVER AGAIN!!! (OFC THIS HAD TO HAPPEN AND BE PAINFUL)
        reduced_dataset = args.quickquick
        if args.experiment_task_id:
            store_name = args.experiment_task_id
            if reduced_dataset:
                store_name = "quickquick_" + store_name
        else:
            store_name = "results"

        # Make sure we have a directory to store the results in, and we don't crash!
        Exp_log_folder = os.path.join(Exp_root, "al_training_test_prob_scores/")
        os.makedirs(Exp_log_folder, exist_ok=True)
        store = laaos.open_file_store(
            store_name+f"_{seed}",
            prefix=Exp_log_folder,
            suffix="",
            truncate=False,
            type_handlers=(laaos.StrEnumHandler(), laaos.ToReprHandler()),
        )
        store["args"] = args.__dict__
        store["cmdline"] = sys.argv[:]

        print("|".join(sys.argv))
        print(args.__dict__)

        acquisition_method: AcquisitionMethod = args.acquisition_method

        use_cuda = not args.no_cuda and torch.cuda.is_available()

        device = torch.device("cuda" if use_cuda else "cpu")

        print(f"Using {device} for computations")

        kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

        dataset: DatasetEnum = args.dataset
        samples_per_class = args.initial_samples_per_class
        validation_set_size = args.validation_set_size
        balanced_test_set = args.balanced_test_set
        balanced_validation_set = args.balanced_validation_set

        experiment_data = get_experiment_data(
            data_source=dataset.get_data_source(),
            num_classes=dataset.num_classes,
            initial_samples=args.initial_samples,
            reduced_dataset=reduced_dataset,
            samples_per_class=samples_per_class,
            validation_set_size=validation_set_size,
            balanced_test_set=balanced_test_set,
            balanced_validation_set=balanced_validation_set,
        )

        test_loader = torch.utils.data.DataLoader(
            experiment_data.test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs
        )

        train_loader = torch.utils.data.DataLoader(
            experiment_data.train_dataset,
            sampler=RandomFixedLengthSampler(experiment_data.train_dataset, args.epoch_samples),
            batch_size=args.batch_size,
            **kwargs,
        )

        available_loader = torch.utils.data.DataLoader(
            experiment_data.available_dataset, batch_size=args.scoring_batch_size, shuffle=False, **kwargs
        )

        validation_loader = torch.utils.data.DataLoader(
            experiment_data.validation_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs
        )

        store["iterations"] = []
        # store wraps the empty list in a storable list, so we need to fetch it separately.
        iterations = store["iterations"]

        store["initial_samples"] = experiment_data.initial_samples

        acquisition_function: AcquisitionFunction = args.type
        max_epochs = args.epochs
        load_model_from_pth = os.path.join(Model_saved_root, "model_final.pth")
        num_epochs = 0
        test_metrics = {"accuracy": 0.60}
        model = mnist_model.BayesianNet(num_classes=10).to(device)
        if load_model_from_pth is not None:
            print(f"\n##### Loading trained model weights saved in {load_model_from_pth} #####\n")
            model.load_state_dict(torch.load(load_model_from_pth))
            
        for iteration in itertools.count(1):
            def desc(name):
                return lambda engine: "%s: %s (%s samples)" % (name, iteration, len(experiment_data.train_dataset))
                
            with ContextStopwatch() as batch_acquisition_stopwatch:
                batch = acquisition_method.acquire_batch(
                    bayesian_model=model,
                    acquisition_function=acquisition_function,
                    available_loader=available_loader,
                    num_classes=dataset.num_classes,
                    k=args.num_inference_samples,
                    b=args.available_sample_k,
                    min_candidates_per_acquired_item=args.min_candidates_per_acquired_item,
                    min_remaining_percentage=args.min_remaining_percentage,
                    initial_percentage=args.initial_percentage,
                    reduce_percentage=args.reduce_percentage,
                    device=device,
                    prob_score_sampling=args.prob_score_sampling,
                )

            original_batch_indices = get_base_indices(experiment_data.available_dataset, batch.indices)
            print(f"Acquiring indices {original_batch_indices}")
            targets = get_targets(experiment_data.available_dataset)
            acquired_targets = [int(targets[index]) for index in batch.indices]
            print(f"Acquiring targets {acquired_targets}")

            iterations.append(
                dict(
                    num_epochs=num_epochs,
                    test_metrics=test_metrics,
                    chosen_targets=acquired_targets,
                    chosen_samples=original_batch_indices,
                    chosen_samples_score=batch.scores,
                    chosen_samples_orignal_score=batch.orignal_scores,
                    train_model_elapsed_time=train_model_stopwatch.elapsed_time if iteration > 1 else 0,
                    batch_acquisition_elapsed_time=batch_acquisition_stopwatch.elapsed_time,
                )
            )

            # update selected dataset and remaining pool set
            experiment_data.active_learning_data.acquire(batch.indices)
            num_acquired_samples = len(experiment_data.active_learning_data.active_dataset) - len(
                experiment_data.initial_samples
            )
            if num_acquired_samples >= args.target_num_acquired_samples:
                print(f"{num_acquired_samples} acquired samples >= {args.target_num_acquired_samples}")
                break
            if test_metrics["accuracy"] >= args.target_accuracy:
                print(f'accuracy {test_metrics["accuracy"]} >= {args.target_accuracy}')
                break
                
            # adjust learning rates:
            # if num_acquired_samples < int(args.target_num_acquired_samples/3): 
            #     print(f"{num_acquired_samples} acquired samples < 1/3 taget acquired size ({int(args.target_num_acquired_samples)})")
            #     lr = args.learning_rate
            #     print(f"Set learning rate to {lr}")
            # elif num_acquired_samples > int(args.target_num_acquired_samples/3) and num_acquired_samples < int(args.target_num_acquired_samples/2):
            #     print(f"{num_acquired_samples} acquired samples in (1/3, 1/2) of taget acquired size ({int(args.target_num_acquired_samples)})")
            #     lr = args.learning_rate * 10
            #     print(f"Set learning rate to {lr}")
            # else:
            #     print(f"{num_acquired_samples} acquired samples > 1/2 taget acquired size ({int(args.target_num_acquired_samples)})")
            #     lr = args.learning_rate * 100
            #     print(f"Set learning rate to {lr}")

            warmup_perc = 0.1
            if num_acquired_samples/args.target_num_acquired_samples < warmup_perc:
                lr = args.learning_rate
                print(f"Set learning rate to {lr} (when {num_acquired_samples}/{args.target_num_acquired_samples} < {warmup_perc})")
            else:
                lr = args.learning_rate * (num_acquired_samples/args.target_num_acquired_samples) * 100
                print(f"Set learning rate to {lr} ( 100 * {args.learning_rate }*{num_acquired_samples}/{args.target_num_acquired_samples})")
            print("Starting fine-tuning!")
            with ContextStopwatch() as train_model_stopwatch:
                early_stopping_patience = args.early_stopping_patience
                num_inference_samples = args.num_inference_samples
                log_interval = args.log_interval
                model, num_epochs, test_metrics = dataset.train_model(
                        train_loader,
                        test_loader,
                        validation_loader,
                        num_inference_samples,
                        max_epochs,
                        early_stopping_patience,
                        desc,
                        log_interval,
                        device,
                        lr=lr,
                        WarmupLRS_flag=False,
                        load_model_from_pth=load_model_from_pth,
                    )

        print("DONE")


if __name__ == "__main__":
    main()
