import torch

from typing import Any, Union, Sequence

from src.simulation.component import Component

################################################################################
# Simulate environmental acoustic distortions in sequence
################################################################################


class Effect(Component):
    """
    Base class for all acoustic simulation effects units. Adds random parameter
    sampling functionality to Component class.
    """
    def __init__(self, compute_grad: bool = True):
        super().__init__(compute_grad)

    def forward(self, x: torch.Tensor):
        raise NotImplementedError()

    def sample_params(self):
        """
        Sample effect parameters to allow for expectation-over-transformation
        """
        raise NotImplementedError()

    @staticmethod
    def parse_range(params: Any, dtype: Any, error_msg: str):
        """
        For real-valued parameters, obtain acceptable range of values from which
        to sample randomly
        """

        # for any sequence, assume endpoints mark range of values
        if isinstance(params, Sequence):
            min_val, max_val = params[0], params[1]

        # if a single value is given, use as both "endpoints"
        elif isinstance(params, dtype):
            min_val = max_val = params

        else:
            raise ValueError(error_msg)

        try:
            assert isinstance(min_val, dtype)
            assert isinstance(max_val, dtype)

        except AssertionError:
            raise ValueError(error_msg)

        return min_val, max_val

