import random
import torch

from src.simulation.effect import Effect

################################################################################
# Simple gain scaling
################################################################################


class Gain(Effect):

    def __init__(self, compute_grad: bool = True, level: any = None):

        super().__init__(compute_grad)

        self.min_level, self.max_level = self.parse_range(
            level,
            float,
            f'Invalid gain {level}'
        )

        self.level = None
        self.sample_params()

    def forward(self, x: torch.Tensor):
        return x * self.level

    def sample_params(self):
        self.level = random.uniform(self.min_level, self.max_level)
