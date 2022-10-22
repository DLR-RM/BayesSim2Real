import enum

from torch import nn as nn

from src.utils.batch_acquisition import compute_acquisition_bag, compute_multi_bald_batch, compute_acquisition_bag_clue
from src.utils.structure import AcquisitionBatch
from src.utils.acquisition_functions import AcquisitionFunction


class AcquisitionMethod(enum.Enum):
    independent = "independent"
    multibald = "multibald"
    clue = "clue"

    def acquire_batch(
        self,
        bayesian_model: nn.Module,
        acquisition_function: AcquisitionFunction,
        available_loader,
        num_classes,
        k,
        b,
        min_candidates_per_acquired_item,
        min_remaining_percentage,
        initial_percentage,
        reduce_percentage,
        device=None,
        prob_score_sampling=False,
    ) -> AcquisitionBatch:
        target_size = max(
            min_candidates_per_acquired_item * b, len(available_loader.dataset) * min_remaining_percentage // 100
        )

        if self == self.independent:
            return compute_acquisition_bag(
                bayesian_model=bayesian_model,
                acquisition_function=acquisition_function,
                num_classes=num_classes,
                k=k,
                b=b,
                initial_percentage=initial_percentage,
                reduce_percentage=reduce_percentage,
                available_loader=available_loader,
                device=device,
                prob_score_sampling=prob_score_sampling,
            )
        elif self == self.multibald:
            return compute_multi_bald_batch(
                bayesian_model=bayesian_model,
                available_loader=available_loader,
                num_classes=num_classes,
                k=k,
                b=b,
                initial_percentage=initial_percentage,
                reduce_percentage=reduce_percentage,
                target_size=target_size,
                device=device,
            )
        
        elif self == self.clue:
            return compute_acquisition_bag_clue(
                bayesian_model=bayesian_model,
                acquisition_function=acquisition_function,
                num_classes=num_classes,
                k=k,
                b=b,
                initial_percentage=initial_percentage,
                reduce_percentage=reduce_percentage,
                available_loader=available_loader,
                device=device,
            )
        else:
            raise NotImplementedError(f"Unknown acquisition method {self}!")
