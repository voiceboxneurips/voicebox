import torch
import torch.nn.functional as F

from src.simulation.component import Component

################################################################################
# Pre-emphasis filter
################################################################################


class PreEmphasis(Component):
    """
    Apply pre-emphasis filter via waveform convolution. Adapted from
    https://github.com/clovaai/voxceleb_trainer/blob/master/utils.py
    """

    def __init__(self, coef: float = 0.97, method: str = 'shift'):
        """
        Initialize filter

        :param coef: pre-emphasis coefficient
        :param method: implementation; must be one of `conv` or `shift`
        """
        super().__init__()
        self.coef = coef

        if method not in ['conv', 'shift', None]:
            raise ValueError(f'Invalid method {method}')
        self.method = method

        # flip filter (cross-correlation --> convolution)
        self.register_buffer(
            'flipped_filter',
            torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, x: torch.Tensor):
        """
        Apply pre-emphasis filter via waveform convolution
        """

        assert x.ndim >= 2  # require batch dimension
        n_batch, signal_length = x.shape[0], x.shape[-1]

        # require channel dimension for convolution
        x = x.reshape(n_batch, -1, signal_length)
        in_channels = x.shape[1]

        if self.method == 'conv':

            # reflect padding to match lengths of in/out
            x = F.pad(x, (1, 0), 'reflect')
            return F.conv1d(
                x,
                self.flipped_filter.repeat(in_channels, 1, 1),
                groups=in_channels
            )

        elif self.method == 'shift':

            return torch.cat(
                [
                    x[..., 0:1],
                    x[..., 1:] - self.coef*x[..., :-1]
                ], dim=-1)

        else:
            return x

