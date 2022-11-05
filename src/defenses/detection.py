import torch
from torch import nn

from typing import Tuple

################################################################################
# Base class for detection defense objects
################################################################################


class Detection(nn.Module):
    """
    Attempt to detect adversarial inputs, typically by observing a difference in
    model response when a transformation is applied to inputs
    """

    def __init__(self, compute_grad: bool = True, threshold: float = 0.0):
        super().__init__()
        self.compute_grad = compute_grad
        self.threshold = threshold

    def forward(self,
                x: torch.Tensor,
                model: nn.Module = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Determine whether input is adversarial, optionally using the response
        of a trained model.

        :param x: audio input, shape (n_batch, n_channels, signal_length)
        :param model: optionally, accept model; some defenses rely on observing
                      a model's response to transformed inputs

        :return: a tuple of tensors holding:
                   * boolean detection flags, shape (n_batch,)
                   * detector scores, shape (n_batch,)
        """
        raise NotImplementedError()
