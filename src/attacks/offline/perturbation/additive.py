import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import random

from typing import Union, Dict

from src.attacks.offline.perturbation.perturbation import Perturbation
from src.utils.plotting import plot_waveform
from src.data import DataProperties

################################################################################
# Apply additive perturbation to waveform audio
################################################################################


class AdditivePerturbation(Perturbation):

    def __init__(self,
                 eps: float,
                 projection_norm: Union[str, int, float],
                 length: Union[int, float] = None,
                 align: str = 'start',
                 loop: bool = False,
                 normalize: bool = False
                 ):

        super().__init__()

        self.eps = eps
        self.projection_norm = projection_norm
        self.normalize = normalize

        assert align in ['start', 'random', 'none', None], \
            f'Invalid alignment; must be one of "start" or "random"'
        self.align = align

        # if length is given as floating-point (time), convert to samples
        if isinstance(length, float):
            length = int(length * DataProperties.get('sample_rate'))
        self.length = length

        # if True, loop perturbation to end of audio
        self.loop = loop

        self.register_parameter(
            "delta", nn.Parameter(torch.zeros(1, 1, self.length)))

    def set_reference(self, x: torch.Tensor):
        """
        Given reference input, initialize perturbation parameters accordingly
        and match input device.

        :param x: reference audio, shape (n_batch, n_channels, signal_length)
        """

        # require batch dimension
        assert x.ndim >= 2, f"Invalid reference audio dimensions {x.shape}"

        # determine whether to match length
        if self.length is None or self.length > x.shape[-1]:
            length = x.shape[-1]
        else:
            length = self.length

        # initialize single-waveform additive perturbation and match reference
        # device
        self.delta = nn.Parameter(
            torch.zeros(1, *x.shape[1:-1], length).to(x.device)
        )

    def forward(self, x: torch.Tensor, *args, **kwargs):
        """
        Apply perturbation to inputs.

        :param x: input audio, shape (n_batch, n_channels, signal_length)
        """

        # do not overwrite incoming audio
        x = x.clone().detach()
        ndims = x.ndim

        # prepare to restore original volume
        peak = torch.max(torch.abs(x), -1)[0].reshape(-1)

        # account for audio shorter than stored perturbation
        orig_len = x.shape[-1]
        x = F.pad(x, (0, max(self.delta.shape[-1] - x.shape[-1], 0)))

        # normalize to apply perturbation
        if self.normalize:
            x = (1.0 / torch.max(torch.abs(x) + 1e-8, dim=-1, keepdim=True)[0]) * x

        # check that input is broadcastable with additive perturbation
        assert self._check_broadcastable(
            x[..., :self.delta.shape[-1]],
            self.delta
        ), \
            f"Cannot broadcast inputs of shape {x.shape} " \
            f"with additive perturbation of shape {self.delta.shape}"

        # determine alignment
        if self.align in ['start', 'none', None]:
            st_idx = 0
        elif self.align == 'random':
            st_idx = random.randrange(
                0,
                x.shape[-1] - self.delta.shape[-1]
            )
        else:
            raise ValueError(f'Invalid alignment {self.align}')

        # if specified, loop to end of input audio
        if not self.loop:
            delta = self.delta
            ed_idx = st_idx + self.delta.shape[-1]
        else:
            remaining = x.shape[-1] - self.delta.shape[-1] - st_idx
            n_loops = math.ceil(remaining / self.delta.shape[-1]) + 1
            ed_idx = x.shape[-1]
            delta = self.delta.repeat(
                (1,) * (self.delta.ndim - 1) + (n_loops,)
            )[..., :ed_idx - st_idx]

        # apply perturbation
        x[..., st_idx:ed_idx] = x[..., st_idx:ed_idx] + delta

        # trim to original length
        x = x[..., :orig_len]

        # peak-normalize to match original input
        if self.normalize:
            factor = peak / torch.max(torch.abs(x), -1)[0].reshape(-1)
            factor = factor.reshape((-1,) + (1,)*(ndims - 1))
            x = x * factor

        return x

    def _visualize_top_level(self) -> Dict[str, torch.Tensor]:
        """
        Visualize top-level (non-recursive) perturbation parameters.

        :return: tag (string) / image (tensor) pairs, stored in a dictionary
        """

        name = self.__class__.__name__

        visualizations = {}

        # plot: additive perturbation
        if self.delta.numel() > 0:

            visualizations = {
                **visualizations,
                f'{name}-parameters': plot_waveform(self.delta)
            }

        # plot: parameter gradients
        if self.delta.grad is not None:

            visualizations = {
                **visualizations,
                f'{name}-gradients': plot_waveform(self.delta.grad)
            }

        return visualizations

    def _project_valid_top_level(self):
        """
        Project top-level (non-recursive) parameters to valid range.
        """

        if self.eps is None:
            return

        # obtained flattened parameters
        flattened = []

        for param in self.parameters(recurse=False):
            if param.requires_grad:
                flattened.append(param.data.detach().flatten())

        flattened = torch.cat(flattened, dim=-1)  # (n_parameters,)

        # project using given p-norm and radius
        if self.projection_norm in [2, float(2), "2"]:
            norm = torch.norm(flattened, p=2) + 1e-20
            factor = torch.min(
                torch.tensor(1.0),
                torch.tensor(self.eps) / norm
            ).view(-1)
            with torch.no_grad():
                flattened.mul_(factor)
        elif self.projection_norm in [float("inf"), "inf"]:
            with torch.no_grad():
                flattened.clamp_(min=-self.eps, max=self.eps)
        else:
            raise ValueError(f'Invalid projection norm {self.projection_norm}')

        # overwrite parameter data
        idx = 0
        for param in self.parameters(recurse=False):
            if param.requires_grad:
                param_length = param.shape.numel()
                data = flattened[idx:idx + param_length].reshape(
                    param.shape
                ).detach()
                param.data = data
                idx += param_length
