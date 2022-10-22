import torch.nn as nn
import torch
import numpy as np

from src.utils.reduced_consistent_mc_sampler import (
    reduced_eval_consistent_bayesian_model,
    reduced_eval_consistent_bayesian_model_emb)
from src.utils.structure import AcquisitionBatch
from src.utils.acquisition_functions import AcquisitionFunction

from src.utils.progress_bar import with_progress_bar
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

import src.utils.joint_entropy.exact as joint_entropy_exact
import src.utils.joint_entropy.sampling as joint_entropy_sampling
import src.utils.torch_utils as torch_utils
import math

def get_top_n(scores: np.ndarray, n):
    top_n = np.argpartition(scores, -n)[-n:]
    return top_n

def compute_acquisition_bag(
    bayesian_model: nn.Module,
    acquisition_function: AcquisitionFunction,
    available_loader,
    num_classes: int,
    k: int,
    b: int,
    initial_percentage: int,
    reduce_percentage: int,
    device=None,
    prob_score_sampling=False,
) -> AcquisitionBatch:
    if acquisition_function != AcquisitionFunction.random:
        result = reduced_eval_consistent_bayesian_model(
            bayesian_model=bayesian_model,
            acquisition_function=acquisition_function,
            num_classes=num_classes,
            k=k,
            initial_percentage=initial_percentage,
            reduce_percentage=reduce_percentage,
            target_size=b,
            available_loader=available_loader,
            device=device,
            prob_score_sampling=prob_score_sampling,
        )

        scores_B = result.scores_B
        subset_split = result.subset_split
        result = None

        top_k_scores, top_k_indices = scores_B.topk(b, largest=True, sorted=True)
        top_k_scores = top_k_scores.numpy()
        # Map our indices to the available_loader dataset.
        top_k_indices = subset_split.get_dataset_indices(top_k_indices.numpy())

        print(f"Acquisition bag: {top_k_indices}")
        print(f"Scores: {top_k_scores}")

        return AcquisitionBatch(top_k_indices, top_k_scores, None)
    
    else:
        picked_indices = torch.randperm(len(available_loader.dataset))[:b].numpy()

        print(f"Acquisition bag: {picked_indices}")

        return AcquisitionBatch(picked_indices, [0.0] * b, None)

def compute_acquisition_bag_clue(
    bayesian_model: nn.Module,
    acquisition_function: AcquisitionFunction,
    available_loader,
    num_classes: int,
    k: int,
    b: int,
    initial_percentage: int,
    reduce_percentage: int,
    device=None,
) -> AcquisitionBatch:
    if acquisition_function != AcquisitionFunction.random:
        result = reduced_eval_consistent_bayesian_model_emb(
            bayesian_model=bayesian_model,
            acquisition_function=acquisition_function,
            num_classes=num_classes,
            k=k,
            initial_percentage=initial_percentage,
            reduce_percentage=reduce_percentage,
            target_size=b,
            available_loader=available_loader,
            device=device,
        )

        scores_B = result.scores_B.numpy()
        embs_B_D = result.embs_B_D.numpy()
        subset_split = result.subset_split
        result = None

        km = KMeans(b)
        km.fit(embs_B_D, sample_weight=scores_B)

        # Find nearest neighbors to inferred centroids
        dists = euclidean_distances(km.cluster_centers_, embs_B_D)
        sort_idxs = dists.argsort(axis=1)
        sel_indicces = []
        ax, rem = 0, b
        while rem > 0:
            inst_idx_list = list(sort_idxs[:, ax][:rem])
            for inst_idx in inst_idx_list:
                sel_indicces.append(inst_idx)
            sel_indicces = list(set(sel_indicces))
            rem = b - len(sel_indicces)
            ax += 1

        # top_k_scores, top_k_indices = scores_B.topk(b, largest=True, sorted=True)
        # top_k_scores = top_k_scores.numpy()

        # Map our indices to the available_loader dataset.
        top_k_indices = subset_split.get_dataset_indices(sel_indicces)

        print(f"Acquisition bag: {top_k_indices}")
        print(f"Scores: {scores_B[sel_indicces]}")

        return AcquisitionBatch(top_k_indices, scores_B[sel_indicces], None)
    else:
        picked_indices = torch.randperm(len(available_loader.dataset))[:b].numpy()

        print(f"Acquisition bag: {picked_indices}")

        return AcquisitionBatch(picked_indices, [0.0] * b, None)

   
compute_multi_bald_bag_multi_bald_batch_size = None

def compute_multi_bald_batch(
    bayesian_model: nn.Module,
    available_loader,
    num_classes,
    k,
    b,
    target_size,
    initial_percentage,
    reduce_percentage,
    device=None,
) -> AcquisitionBatch:
    result = reduced_eval_consistent_bayesian_model(
        bayesian_model=bayesian_model,
        acquisition_function=AcquisitionFunction.bald,
        num_classes=num_classes,
        k=k,
        initial_percentage=initial_percentage,
        reduce_percentage=reduce_percentage,
        target_size=target_size,
        available_loader=available_loader,
        device=device,
    )

    subset_split = result.subset_split

    partial_multi_bald_B = result.scores_B
    # Now we can compute the conditional entropy
    conditional_entropies_B = joint_entropy_exact.batch_conditional_entropy_B(result.logits_B_K_C)

    # We turn the logits into probabilities.
    probs_B_K_C = result.logits_B_K_C.exp_()

    # Don't need the result anymore.
    result = None

    torch_utils.gc_cuda()
    # torch_utils.cuda_meminfo()

    with torch.no_grad():
        num_samples_per_ws = 40000 // k
        num_samples = num_samples_per_ws * k

        if device.type == "cuda":
            # KC_memory = k*num_classes*8
            sample_MK_memory = num_samples * k * 8
            MC_memory = num_samples * num_classes * 8
            copy_buffer_memory = 256 * num_samples * num_classes * 8
            slack_memory = 2 * 2 ** 30
            multi_bald_batch_size = (
                torch_utils.get_cuda_available_memory() - (sample_MK_memory + copy_buffer_memory + slack_memory)
            ) // MC_memory

            global compute_multi_bald_bag_multi_bald_batch_size
            if compute_multi_bald_bag_multi_bald_batch_size != multi_bald_batch_size:
                compute_multi_bald_bag_multi_bald_batch_size = multi_bald_batch_size
                print(f"New compute_multi_bald_bag_multi_bald_batch_size = {multi_bald_batch_size}")
        else:
            multi_bald_batch_size = 16

        subset_acquisition_bag = []
        global_acquisition_bag = []
        acquisition_bag_scores = []

        # We use this for early-out in the b==0 case.
        MIN_SPREAD = 0.1

        if b == 0:
            b = 100
            early_out = True
        else:
            early_out = False

        prev_joint_probs_M_K = None
        prev_samples_M_K = None
        for i in range(b):
            torch_utils.gc_cuda()

            if i > 0:
                # Compute the joint entropy
                joint_entropies_B = torch.empty((len(probs_B_K_C),), dtype=torch.float64)

                exact_samples = num_classes ** i
                if exact_samples <= num_samples:
                    prev_joint_probs_M_K = joint_entropy_exact.joint_probs_M_K(
                        probs_B_K_C[subset_acquisition_bag[-1]][None].to(device),
                        prev_joint_probs_M_K=prev_joint_probs_M_K,
                    )

                    # torch_utils.cuda_meminfo()
                    batch_exact_joint_entropy(
                        probs_B_K_C, prev_joint_probs_M_K, multi_bald_batch_size, device, joint_entropies_B
                    )
                else:
                    if prev_joint_probs_M_K is not None:
                        prev_joint_probs_M_K = None
                        torch_utils.gc_cuda()

                    # Gather new traces for the new subset_acquisition_bag.
                    prev_samples_M_K = joint_entropy_sampling.sample_M_K(
                        probs_B_K_C[subset_acquisition_bag].to(device), S=num_samples_per_ws
                    )

                    # torch_utils.cuda_meminfo()
                    for joint_entropies_b, probs_b_K_C in with_progress_bar(
                        torch_utils.split_tensors(joint_entropies_B, probs_B_K_C, multi_bald_batch_size),
                        unit_scale=multi_bald_batch_size,
                    ):
                        joint_entropies_b.copy_(
                            joint_entropy_sampling.batch(probs_b_K_C.to(device), prev_samples_M_K), non_blocking=True
                        )

                        # torch_utils.cuda_meminfo()

                    prev_samples_M_K = None
                    torch_utils.gc_cuda()

                partial_multi_bald_B = joint_entropies_B - conditional_entropies_B
                joint_entropies_B = None

            # Don't allow reselection
            partial_multi_bald_B[subset_acquisition_bag] = -math.inf

            winner_index = partial_multi_bald_B.argmax().item()

            # Actual MultiBALD is:
            actual_multi_bald_B = partial_multi_bald_B[winner_index] - torch.sum(
                conditional_entropies_B[subset_acquisition_bag]
            )
            actual_multi_bald_B = actual_multi_bald_B.item()

            print(f"Actual MultiBALD: {actual_multi_bald_B}")

            # If we early out, we don't take the point that triggers the early out.
            # Only allow early-out after acquiring at least 1 sample.
            if early_out and i > 1:
                current_spread = actual_multi_bald_B[winner_index] - actual_multi_bald_B.median()
                if current_spread < MIN_SPREAD:
                    print("Early out")
                    break

            acquisition_bag_scores.append(actual_multi_bald_B)

            subset_acquisition_bag.append(winner_index)
            # We need to map the index back to the actual dataset.
            global_acquisition_bag.append(subset_split.get_dataset_indices([winner_index]).item())

            print(f"Acquisition bag: {sorted(global_acquisition_bag)}")

    return AcquisitionBatch(global_acquisition_bag, acquisition_bag_scores, None)


def batch_exact_joint_entropy(probs_B_K_C, prev_joint_probs_M_K, chunk_size, device, out_joint_entropies_B):
    """This one switches between devices, too."""
    for joint_entropies_b, probs_b_K_C in with_progress_bar(
        torch_utils.split_tensors(out_joint_entropies_B, probs_B_K_C, chunk_size), unit_scale=chunk_size
    ):
        joint_entropies_b.copy_(
            joint_entropy_exact.batch(probs_b_K_C.to(device), prev_joint_probs_M_K), non_blocking=True
        )

    return joint_entropies_b


def batch_exact_joint_entropy_logits(logits_B_K_C, prev_joint_probs_M_K, chunk_size, device, out_joint_entropies_B):
    """This one switches between devices, too."""
    for joint_entropies_b, logits_b_K_C in with_progress_bar(
        torch_utils.split_tensors(out_joint_entropies_B, logits_B_K_C, chunk_size), unit_scale=chunk_size
    ):
        joint_entropies_b.copy_(
            joint_entropy_exact.batch(logits_b_K_C.to(device).exp(), prev_joint_probs_M_K), non_blocking=True
        )

    return joint_entropies_b
