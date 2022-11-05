import random
import torch

from src.simulation.effect import Effect

################################################################################
# Random time-domain offset
################################################################################


class Offset(Effect):

    def __init__(self, compute_grad: bool = True, length: any = None):
        """
        Shift audio and trim/zero-pad to maintain length

        :param compute_grad: if False, use straight-through gradient estimator
        :param length: offset length in seconds; sign indicates direction
        """
        super().__init__(compute_grad)

        self.min_length, self.max_length = self.parse_range(
            length,
            float,
            f'Invalid offset length {length}'
        )

        self.length = None
        self.sample_params()

    def forward(self, x: torch.Tensor):
        shifted = torch.roll(x, shifts=self.length, dims=-1)
        if self.length >= 0:
            shifted[..., :self.length] = 0
        else:
            shifted[..., self.length:] = 0
        return shifted

    def sample_params(self):
        self.length = round(
            random.uniform(self.min_length, self.max_length) * self.sample_rate
        )
