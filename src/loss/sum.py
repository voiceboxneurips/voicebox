import torch
import torch.nn as nn
from src.loss.auxiliary import AuxiliaryLoss

from typing import List

################################################################################
# Combine multiple auxiliary losses with a weighted sum
################################################################################


class SumLoss(AuxiliaryLoss):
    """
    Calculates a weighted sum of auxiliary loss functions
    """
    def __init__(self, reduction: str = 'none'):
        super().__init__(reduction=reduction)

        self._loss_functions: nn.ModuleList[nn.Module] = nn.ModuleList()
        self._loss_weights: List[float] = []

    def add_loss_function(self,
                          loss: AuxiliaryLoss,
                          weight: float) -> AuxiliaryLoss:
        """
        Adds loss function to `_loss_functions` with `_loss_weights`
        """

        assert loss.reduction == 'none', \
            "Losses must provide unreduced batch values"

        self._loss_functions.append(loss)
        self._loss_weights.append(weight)

        return self

    def _compute_loss(self,
                      x: torch.Tensor,
                      x_ref: torch.Tensor = None) -> torch.Tensor:
        """
        Compute weighted sum over all losses
        """

        # require batch dimension
        assert x.ndim >= 2
        n_batch = x.shape[0]

        # compute unreduced total batch loss
        loss_total = torch.zeros(n_batch).to(x.device)

        for loss, weight in zip(self._loss_functions, self._loss_weights):
            loss_total += weight * loss(x, x_ref)
        return loss_total

    def set_reference(self, x_ref: torch.Tensor):
        """
        Compute and cache reference representation(s) for all stored losses
        """
        for loss in self._loss_functions:
            loss.set_reference(x_ref)
