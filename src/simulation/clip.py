import random
import torch

from src.simulation.effect import Effect

################################################################################
# Perform simple clipping at waveform
################################################################################


class Clip(Effect):

    def __init__(self,
                 compute_grad: bool = True,
                 scale: any = 1.0):
        super().__init__(compute_grad)

        # parse valid range of clipping scale parameter
        self.min_scale, self.max_scale = self.parse_range(
            scale,
            float,
            f'Invalid clipping scale {scale}'
        )

        assert 0 <= scale <= self.scale

        self.clip_scale = None
        self.sample_params()

    def forward(self, x: torch.Tensor):
        return torch.clamp(x, min=-self.clip_scale, max=self.clip_scale)

    def sample_params(self):
        self.clip_scale = random.uniform(self.min_scale, self.max_scale)


