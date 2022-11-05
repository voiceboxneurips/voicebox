import random
import torch

from src.simulation.effect import Effect

################################################################################
# Simulate simple quantization distortions
################################################################################


class Quantize(Effect):

    def __init__(self, bits: any = 8):
        super().__init__(compute_grad=False)

        self.min_bits, self.max_bits = self.parse_range(
            bits,
            int,
            f'Invalid bit depth {bits}'
        )
        self.bits = None
        self.sample_params()

    def forward(self, x: torch.Tensor):

        # rescale full range to -2^(bits - 1), 2^(bits - 1)
        scale_factor = 2 ** (self.bits - 1)
        x_scaled = x * scale_factor / self.scale
        x_quant = torch.round(x_scaled)
        return x_quant * self.scale / scale_factor

    def sample_params(self):
        self.bits = round(
            random.uniform(self.min_bits, self.max_bits)
        )
