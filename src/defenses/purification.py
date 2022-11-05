import torch
import torch.nn as nn

################################################################################
# Base class for purification defense objects
################################################################################


class Purification(nn.Module):
    """
    Attempt to 'purify' adversarial inputs by applying a distortion (e.g.
    filter, compression)
    """

    def __init__(self, compute_grad: bool = True):
        super().__init__()
        self.compute_grad = compute_grad

    def forward(self, x: torch.Tensor):
        raise NotImplementedError()
