import torch
import torch.nn as nn

from typing import Iterable

from src.simulation.component import Component

################################################################################
# Wrap preprocessing stages and apply sequentially
################################################################################


class Preprocessor(nn.Module):
    """
    Wrapper for sequential application of preprocessing stages. Allows for
    straight-through gradient estimation. Because random parameter sampling is
    not required, all modules are only required to be Component objects
    """
    def __init__(self, *args):
        super().__init__()

        stages = []

        if len(args) == 1 and isinstance(args[0], Iterable):
            for stage in args[0]:
                assert isinstance(stage, Component), \
                    "Arguments must be Component objects"
                stages.append(stage)
        else:
            for stage in args:
                assert isinstance(stage, Component), \
                    "Arguments must be Component objects"
                stages.append(stage)

        self.stages = nn.ModuleList(stages)

    def forward(self, x: torch.Tensor):

        # apply in sequence
        for stage in self.stages:

            if stage.compute_grad:
                x = stage(x)

            else:
                # allow straight-through gradient estimation on backward pass
                output = stage(x)
                x = x + (output-x).detach()

        return x

