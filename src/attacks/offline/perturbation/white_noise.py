import torch
import torch.nn as nn

import math
import random

from typing import Union, Dict

from src.attacks.offline.perturbation.perturbation import Perturbation
from src.data import DataProperties

################################################################################
# Apply additive perturbation to waveform audio
################################################################################


class WhiteNoisePerturbation(Perturbation):

    def __init__(self, snr: float = 0.0):

        super().__init__()
        self.snr = nn.Parameter(torch.as_tensor([snr], dtype=torch.float32))

    def set_reference(self, x: torch.Tensor):
        """
        Given reference input, initialize perturbation parameters accordingly
        and match input device.

        :param x: reference audio, shape (n_batch, n_channels, signal_length)
        """
        self.snr = self.snr.to(x.device)

    def set_snr(self, snr: float):
        self.snr.fill_(snr)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        """
        Apply perturbation to inputs.

        :param x: input audio, shape (n_batch, n_channels, signal_length)
        """

        # do not overwrite incoming audio
        x = x.clone().detach()

        # numerical stability
        eps = 1e-8

        # white noise
        noise = torch.randn_like(x)

        # scale noise level to stored SNR
        noise_db = 10 * torch.log10(
            torch.mean(torch.square(noise), dim=-1, keepdims=True) + eps
        )

        signal_db = 10 * torch.log10(
            torch.mean(torch.square(x), dim=-1, keepdims=True) + eps
        )

        scale = torch.sqrt(
            torch.pow(10, (signal_db - noise_db - self.snr) / 10)
        )

        return (scale * noise) + x

    def _visualize_top_level(self) -> Dict[str, torch.Tensor]:
        """
        Visualize top-level (non-recursive) perturbation parameters.

        :return: tag (string) / image (tensor) pairs, stored in a dictionary
        """

        visualizations = {}
        return visualizations

    def _project_valid_top_level(self):
        """
        Project top-level (non-recursive) parameters to valid range.
        """
        pass
