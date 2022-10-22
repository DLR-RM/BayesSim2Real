# batch_bald utility functions adapted from https://github.com/BlackHC/BatchBALD
# and https://github.com/BlackHC/batchbald_redux

import math
from dataclasses import dataclass
from typing import List

import torch
from toma import toma
from tqdm.auto import tqdm

from src.active_learning_src.utils.batch_bald import joint_entropy

def compute_conditional_entropy_prob(probs_N_K_C: torch.Tensor) -> torch.Tensor:
    N, K, C = probs_N_K_C.shape

    entropies_N = torch.empty(N, dtype=torch.double)

    pbar = tqdm(total=N, desc="Conditional Entropy", leave=False)

    @toma.execute.chunked(probs_N_K_C, 1024)
    def compute(probs_n_K_C, start: int, end: int):
        nats_n_K_C = probs_n_K_C * torch.log(probs_n_K_C)
        nats_n_K_C[probs_n_K_C == 0] = 0.0

        entropies_N[start:end].copy_(-torch.sum(nats_n_K_C, dim=(1, 2)) / K)
        pbar.update(end - start)

    pbar.close()

    return entropies_N


def compute_entropy_prob(probs_N_K_C: torch.Tensor) -> torch.Tensor:
    N, K, C = probs_N_K_C.shape

    entropies_N = torch.empty(N, dtype=torch.double)

    pbar = tqdm(total=N, desc="Entropy", leave=False)

    @toma.execute.chunked(probs_N_K_C, 1024)
    def compute(probs_n_K_C, start: int, end: int):
        mean_probs_n_C = probs_n_K_C.mean(dim=1)
        nats_n_C = mean_probs_n_C * torch.log(mean_probs_n_C)
        nats_n_C[mean_probs_n_C == 0] = 0.0

        entropies_N[start:end].copy_(-torch.sum(nats_n_C, dim=1))
        pbar.update(end - start)

    pbar.close()

    return entropies_N

def compute_conditional_entropy(log_probs_N_K_C: torch.Tensor) -> torch.Tensor:
    N, K, C = log_probs_N_K_C.shape

    entropies_N = torch.empty(N, dtype=torch.double)

    pbar = tqdm(total=N, desc="Conditional Entropy", leave=False)

    @toma.execute.chunked(log_probs_N_K_C, 1024)
    def compute(log_probs_n_K_C, start: int, end: int):
        nats_n_K_C = log_probs_n_K_C * torch.exp(log_probs_n_K_C)

        entropies_N[start:end].copy_(-torch.sum(nats_n_K_C, dim=(1, 2)) / K)
        pbar.update(end - start)

    pbar.close()

    return entropies_N


def compute_entropy(log_probs_N_K_C: torch.Tensor) -> torch.Tensor:
    N, K, C = log_probs_N_K_C.shape

    entropies_N = torch.empty(N, dtype=torch.double)

    pbar = tqdm(total=N, desc="Entropy", leave=False)

    @toma.execute.chunked(log_probs_N_K_C, 1024)
    def compute(log_probs_n_K_C, start: int, end: int):
        mean_log_probs_n_C = torch.logsumexp(log_probs_n_K_C, dim=1) - math.log(K)
        nats_n_C = mean_log_probs_n_C * torch.exp(mean_log_probs_n_C)

        entropies_N[start:end].copy_(-torch.sum(nats_n_C, dim=1))
        pbar.update(end - start)

    pbar.close()

    return entropies_N


def get_batchbald_batch(pool_inst_probs_N_K_C, 
                        num_select, 
                        img_idx_N,
                        num_samples=10000, 
                        dtype=None, 
                        device=None):
    log_probs_N_K_C = torch.log(pool_inst_probs_N_K_C)
    N, K, C = log_probs_N_K_C.shape

    num_select = min(num_select, N)

    candidate_indices = []
    candidate_scores = []
    selected_instances_id_list = []

    if num_select == 0:
        return [] # CandidateBatch(candidate_scores, candidate_indices)

    conditional_entropies_N = compute_conditional_entropy(log_probs_N_K_C)

    batch_joint_entropy = joint_entropy.DynamicJointEntropy(
        num_samples, num_select - 1, K, C, dtype=dtype, device=device
    )

    # We always keep these on the CPU.
    scores_N = torch.empty(N, dtype=torch.double, pin_memory=torch.cuda.is_available())

    for i in tqdm(range(num_select), desc="BatchBALD", leave=False):
        if i > 0:
            latest_index = candidate_indices[-1]
            batch_joint_entropy.add_variables(log_probs_N_K_C[latest_index : latest_index + 1])

        shared_conditinal_entropies = conditional_entropies_N[candidate_indices].sum()

        batch_joint_entropy.compute_batch(log_probs_N_K_C, output_entropies_B=scores_N)

        scores_N -= conditional_entropies_N + shared_conditinal_entropies
        scores_N[candidate_indices] = -float("inf")

        # to avoid select instances in the same image
        max_idx_list = torch.argsort(scores_N, dim=0, descending=True)
        max_idx = 0
        found_counter = 0 # counter to record the number of candidate idx that is not in the same image of the current one
        if len(candidate_indices) > 0:
            while found_counter < len(candidate_indices):
                selected_idx= max_idx_list[max_idx]
                found_counter = 0
                for can_idx in candidate_indices:
                    # if the current selected index is found to be in the same image with any selected index
                    # then go to the next maximum index
                    if img_idx_N[can_idx] == img_idx_N[selected_idx]:
                        max_idx += 1
                        break
                    else:
                        found_counter += 1

        candidate_score, candidate_index = scores_N[max_idx_list[max_idx]], max_idx_list[max_idx]

        candidate_indices.append(candidate_index.item())
        candidate_scores.append(candidate_score.item())

    return candidate_indices, candidate_scores # CandidateBatch(candidate_scores, candidate_indices)