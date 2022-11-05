import random
import torch

from src.simulation.effect import Effect

################################################################################
# Random time-domain dropout
################################################################################


class Dropout(Effect):

    def __init__(self, compute_grad: bool = True, rate: any = None):

        super().__init__(compute_grad)

        self.min_rate, self.max_rate = self.parse_range(
            rate,
            float,
            f'Invalid signal dropout rate {rate}'
        )

        # store waveform mask as buffer to allow device movement
        self.register_buffer("mask", torch.zeros(1, dtype=torch.float32))
        self.sample_params()

    def forward(self, x: torch.Tensor):
        return self.mask.clone().to(x) * x

    def sample_params(self):
        """
        Sample dropout rate uniformly and apply random dropout
        """
        rate = random.uniform(self.min_rate, self.max_rate)
        idx = torch.randperm(self.signal_length
                             )[:round(rate * self.signal_length)]
        self.mask = torch.ones(self.signal_length).to(self.mask)
        self.mask[..., idx] = 0
