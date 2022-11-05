import torch
import torch.nn as nn

from src.data import DataProperties

################################################################################
# Base class for differentiable audio-processing units
################################################################################


class Component(nn.Module):
    """
    Base class for differentiable audio-processing units
    """
    def __init__(self, compute_grad: bool = True):
        super().__init__()
        self.compute_grad = compute_grad

        # fetch persistent data properties
        self.sample_rate, self.scale, self.signal_length = DataProperties.get(
            'sample_rate',
            'scale',
            'signal_length'
        )

    def forward(self, x: torch.Tensor):
        raise NotImplementedError()
