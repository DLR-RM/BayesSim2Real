from dataclasses import dataclass

from src.utils.progress_bar import with_progress_bar
from torch import nn as nn
from torch.utils import data

from src.utils.structure import ActiveLearningData
import src.models.mc_dropout as mc_dropout
import torch

import src.utils.torch_utils as torch_utils
from src.utils.acquisition_functions import AcquisitionFunction


@dataclass
class SubsetEvalResults:
    subset_split: ActiveLearningData
    subset_dataloader: data.DataLoader
    scores_B: torch.Tensor
    logits_B_K_C: torch.Tensor
    embs_B_D:torch.Tensor=None


reduced_eval_consistent_bayesian_model_cuda_chunk_size = 512
def reduced_eval_consistent_bayesian_model_emb(
    bayesian_model: mc_dropout.BayesianModule,
    acquisition_function: AcquisitionFunction,
    num_classes: int,
    k: int,
    initial_percentage: int,
    reduce_percentage: int,
    target_size: int,
    available_loader,
    device=None,
) -> SubsetEvalResults:
    """Performs a scoring step with k inference samples while reducing the dataset to at most min_remaining_percentage.

    Before computing anything at all the initial available dataset is randomly culled to initial_percentage.

    Every `chunk_size` inferences BALD is recomputed and the bottom `reduce_percentage` samples are dropped."""
    global reduced_eval_consistent_bayesian_model_cuda_chunk_size

    # TODO: ActiveLearningData should be renamed to be a more modular SplitDataset.
    # Here, we need to use available_dataset because it allows us to easily recover the original indices.

    # We start with all data in the acquired data.
    subset_split = ActiveLearningData(available_loader.dataset)
    initial_length = len(available_loader.dataset)

    initial_split_length = initial_length * initial_percentage // 100

    # By acquiring [initial_split_length:], we make the tail unavailable.
    subset_split.acquire(torch.randperm(initial_length)[initial_split_length:])

    subset_dataloader = data.DataLoader(
        subset_split.available_dataset, shuffle=False, batch_size=available_loader.batch_size
    )

    print(f"Scoring subset of {len(subset_dataloader.dataset)} items:")

    # We're done with available_loader in this function.
    available_loader = None

    with torch.no_grad():
        B = len(subset_split.available_dataset)
        C = num_classes
        D = bayesian_model.pen_emb_dim

        # We stay on the CPU.
        logits_B_K_C = None
        embs_B_K_D = None

        k_lower = 0
        torch_utils.gc_cuda()
        chunk_size = reduced_eval_consistent_bayesian_model_cuda_chunk_size if device.type == "cuda" else 32
        while k_lower < k:
            try:
                k_upper = min(k_lower + chunk_size, k)

                old_logit_B_K_C = logits_B_K_C
                old_emb_B_K_D = embs_B_K_D
                # This also stays on the CPU.
                logits_B_K_C = torch.empty((B, k_upper, C), dtype=torch.float64)
                embs_B_K_D = torch.empty((B, k_upper, D), dtype=torch.float64)

                # Copy the old data over.
                if k_lower > 0:
                    logits_B_K_C[:, 0:k_lower, :].copy_(old_logit_B_K_C)
                    embs_B_K_D[:, 0:k_lower, :].copy_(old_emb_B_K_D)
                    old_logit_B_K_C = None
                    old_emb_B_K_D = None

                # This resets the dropout masks.
                bayesian_model.eval()

                for i, (batch, _) in enumerate(
                    with_progress_bar(subset_dataloader, unit_scale=subset_dataloader.batch_size)
                ):
                    lower = i * subset_dataloader.batch_size
                    upper = min(lower + subset_dataloader.batch_size, B)

                    batch = batch.to(device)
                    # batch_size x ws x classes
                    
                    mc_output_B_K_C, mc_feat_B_K_D = bayesian_model(batch, k_upper - k_lower, return_pen_emb=True)
                    
                    logits_B_K_C[lower:upper, k_lower:k_upper].copy_(mc_output_B_K_C.double(), non_blocking=True)
                    embs_B_K_D[lower:upper, k_lower:k_upper].copy_(mc_feat_B_K_D.double(), non_blocking=True)

            except RuntimeError as exception:
                if torch_utils.should_reduce_batch_size(exception):
                    if chunk_size <= 1:
                        raise
                    chunk_size = chunk_size // 2
                    print(f"New reduced_eval_consistent_bayesian_model_cuda_chunk_size={chunk_size} ({exception})")
                    reduced_eval_consistent_bayesian_model_cuda_chunk_size = chunk_size

                    torch_utils.gc_cuda()
                else:
                    raise
            else:
                if k_upper == k:
                    next_size = target_size
                elif k_upper < 50:
                    next_size = B
                else:
                    next_size = max(target_size, B * (100 - reduce_percentage) // 100)

                # Compute the score if it's needed: we are going to reduce the dataset or we're in the last iteration.
                if next_size < B or k_upper == k:
                    scores_B = acquisition_function.compute_scores(
                        logits_B_K_C, available_loader=subset_dataloader, device=device
                    )
                    embs_B_D = torch.mean(embs_B_K_D, dim=1, keepdim=False)
                else:
                    scores_B = None

                k_lower += chunk_size

    return SubsetEvalResults(
        subset_split=subset_split, 
        subset_dataloader=subset_dataloader, 
        scores_B=scores_B, 
        logits_B_K_C=logits_B_K_C,
        embs_B_D=embs_B_D
    )

def reduced_eval_consistent_bayesian_model(
    bayesian_model: mc_dropout.BayesianModule,
    acquisition_function: AcquisitionFunction,
    num_classes: int,
    k: int,
    initial_percentage: int,
    reduce_percentage: int,
    target_size: int,
    available_loader,
    device=None,
    prob_score_sampling=False,
) -> SubsetEvalResults:
    """Performs a scoring step with k inference samples while reducing the dataset to at most min_remaining_percentage.

    Before computing anything at all the initial available dataset is randomly culled to initial_percentage.

    Every `chunk_size` inferences BALD is recomputed and the bottom `reduce_percentage` samples are dropped."""
    global reduced_eval_consistent_bayesian_model_cuda_chunk_size

    # TODO: ActiveLearningData should be renamed to be a more modular SplitDataset.
    # Here, we need to use available_dataset because it allows us to easily recover the original indices.

    # We start with all data in the acquired data.
    subset_split = ActiveLearningData(available_loader.dataset)
    initial_length = len(available_loader.dataset)

    initial_split_length = initial_length * initial_percentage // 100

    # By acquiring [initial_split_length:], we make the tail unavailable.
    subset_split.acquire(torch.randperm(initial_length)[initial_split_length:])

    subset_dataloader = data.DataLoader(
        subset_split.available_dataset, shuffle=False, batch_size=available_loader.batch_size
    )

    print(f"Scoring subset of {len(subset_dataloader.dataset)} items:")

    # We're done with available_loader in this function.
    available_loader = None

    with torch.no_grad():
        B = len(subset_split.available_dataset)
        C = num_classes

        # We stay on the CPU.
        logits_B_K_C = None

        k_lower = 0
        torch_utils.gc_cuda()
        chunk_size = reduced_eval_consistent_bayesian_model_cuda_chunk_size if device.type == "cuda" else 32
        while k_lower < k:
            try:
                k_upper = min(k_lower + chunk_size, k)

                old_logit_B_K_C = logits_B_K_C
                # This also stays on the CPU.
                logits_B_K_C = torch.empty((B, k_upper, C), dtype=torch.float64)

                # Copy the old data over.
                if k_lower > 0:
                    logits_B_K_C[:, 0:k_lower, :].copy_(old_logit_B_K_C)
                    old_logit_B_K_C = None

                # This resets the dropout masks.
                bayesian_model.eval()

                for i, (batch, _) in enumerate(
                    with_progress_bar(subset_dataloader, unit_scale=subset_dataloader.batch_size)
                ):
                    lower = i * subset_dataloader.batch_size
                    upper = min(lower + subset_dataloader.batch_size, B)

                    batch = batch.to(device)
                    # batch_size x ws x classes
                    mc_output_B_K_C = bayesian_model(batch, k_upper - k_lower)
                    logits_B_K_C[lower:upper, k_lower:k_upper].copy_(mc_output_B_K_C.double(), non_blocking=True)

            except RuntimeError as exception:
                if torch_utils.should_reduce_batch_size(exception):
                    if chunk_size <= 1:
                        raise
                    chunk_size = chunk_size // 2
                    print(f"New reduced_eval_consistent_bayesian_model_cuda_chunk_size={chunk_size} ({exception})")
                    reduced_eval_consistent_bayesian_model_cuda_chunk_size = chunk_size

                    torch_utils.gc_cuda()
                else:
                    raise
            else:
                if k_upper == k:
                    next_size = target_size
                elif k_upper < 50:
                    next_size = B
                else:
                    next_size = max(target_size, B * (100 - reduce_percentage) // 100)

                # Compute the score if it's needed: we are going to reduce the dataset or we're in the last iteration.
                if next_size < B or k_upper == k:
                    scores_B = acquisition_function.compute_scores(
                        logits_B_K_C, available_loader=subset_dataloader, device=device
                    )
                else:
                    scores_B = None

                if next_size < B:
                    print("Reducing size", next_size)
                    if prob_score_sampling:
                        scores_B_porob = torch.softmax(scores_B, 0)
                        new_indices = scores_B_porob.multinomial(num_samples=next_size, replacement=False)
                        scores_B = scores_B_porob[new_indices].clone().detach()
                    else:
                        sorted_indices = torch.argsort(scores_B, descending=True)
                        new_indices = torch.sort(sorted_indices[:next_size], descending=False)[0]
                        scores_B = scores_B[new_indices].clone().detach()

                    B = next_size
                    logits_B_K_C = logits_B_K_C[new_indices]
                    if k_upper == k:
                        logits_B_K_C = logits_B_K_C.clone().detach()

                    # Acquire all the low scorers
                    subset_split.acquire(new_indices)

                k_lower += chunk_size

    return SubsetEvalResults(
        subset_split=subset_split, subset_dataloader=subset_dataloader, scores_B=scores_B, logits_B_K_C=logits_B_K_C
    )
