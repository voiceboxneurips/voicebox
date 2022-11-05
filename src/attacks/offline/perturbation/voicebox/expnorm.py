import math

import torch
import torch.nn as nn

from src.data import DataProperties

################################################################################
# Exponential unit normalization module
################################################################################


class ExponentialUnitNorm(nn.Module):
    """Unit-normalize magnitude spectrogram"""

    def __init__(self,
                 decay: float,
                 hop_size: int,
                 n_freq: int,
                 eps: float = 1e-14):
        """
        Perform exponential unit normalization on magnitude spectrogram

        Parameters
        ----------
        decay (float):

        hop_size (int):

        n_freq (int):

        eps (float):
        """

        super().__init__()

        # compute exponential decay factor
        self.alpha = self._get_norm_alpha(
            DataProperties.get('sample_rate'),
            hop_size,
            decay
        )
        self.eps = eps
        self.n_freq = n_freq
        self.init_state: torch.Tensor

        # initialize per-band states for unit norm calculation
        self.reset()

    @staticmethod
    def _get_norm_alpha(sr: int, hop_size: int, decay: float):
        """
        Compute exponential decay factor `alpha` for a given decay window size
        in seconds
        """
        dt = hop_size / sr
        a_ = math.exp(-dt / decay)

        precision = 3
        a = 1.0

        while a >= 1.0:
            a = round(a_, precision)
            precision += 1

        return a

    def reset(self):
        """(Re)-initialize stored state"""
        s = torch.linspace(0.001, 0.0001, self.n_freq).view(
            1, self.n_freq
        )  # broadcast with (n_batch, 1, n_frames, n_freq, 2)
        self.register_buffer("init_state", s)

    def forward(self, x: torch.Tensor):
        """
        Perform exponential unit normalization on magnitude spectrogram

        Parameters
        ----------
        x (Tensor): shape (n_batch, n_freq, n_frames)

        Returns
        -------
        normalized (Tensor): shape (n_batch, n_freq, n_frames)
        """
        x_abs = x.clamp_min(1e-10).sqrt()

        n_batch, n_freq, n_frames = x.shape
        assert n_freq == self.n_freq

        state = self.init_state.clone().expand(
            n_batch, n_freq)  # (n_batch, n_freq)

        out_states = []
        for f in range(n_frames):
            state = x_abs[:, :, f] * (1 - self.alpha) + state * self.alpha
            out_states.append(state)

        return x / torch.stack(out_states, 2).sqrt()
