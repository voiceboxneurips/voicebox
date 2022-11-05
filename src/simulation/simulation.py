import torch
import torch.nn as nn

from typing import Iterable

from src.simulation.effect import Effect

################################################################################
# Wrap effects units to apply in sequence
################################################################################


class Simulation(nn.Module):
    """
    Wrapper for sequential application of effects units. Allows for straight-
    through gradient estimation and random effect parameter sampling.
    """
    def __init__(self, *args):
        super().__init__()

        effects = []

        if len(args) == 1 and isinstance(args[0], Iterable):
            for effect in args[0]:
                assert isinstance(effect, Effect), \
                    "Arguments must be Effect objects"
                effects.append(effect)
        else:
            for effect in args:
                assert isinstance(effect, Effect), \
                    "Arguments must be Effect objects"
                effects.append(effect)

        self.effects = nn.ModuleList(effects)

    def forward(self, x: torch.Tensor):

        for effect in self.effects:

            if effect.compute_grad:
                x = effect(x)

            else:
                # allow straight-through gradient estimation on backward pass
                output = effect(x)
                x = x + (output-x).detach()

        return x

    def sample_params(self):

        for effect in self.effects:
            effect.sample_params()
