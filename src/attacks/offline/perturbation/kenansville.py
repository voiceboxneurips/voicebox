import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft

import math
import random

import numpy as np

from typing import Union, Dict

from src.attacks.offline.perturbation.perturbation import Perturbation
from src.data import DataProperties

################################################################################
# Remove content of spectral bins below energy threshold
################################################################################


class KenansvillePerturbation(Perturbation):

    def __init__(self,
                 threshold_db: float = 100.0,
                 win_length: int = 2048,
                 win_type: str = 'hann'
                 ):

        super().__init__()

        if win_type not in ['rectangular', 'triangular', 'hann']:
            raise ValueError(f'Invalid window type {win_type}')

        self.threshold_db = nn.Parameter(torch.as_tensor([threshold_db]))
        self.win_length = win_length
        self.win_type = win_type

        # determine hop length from window function
        if self.win_type == 'rectangular':  # non-overlapping frames
            self.hop_length = self.win_length
        else:
            self.hop_length = self.win_length // 2

    @staticmethod
    def _get_win_func(win_type: str):
        if win_type == 'rectangular':
            return lambda m: torch.ones(m)
        elif win_type == 'triangular':
            return lambda m: torch.as_tensor(np.bartlett(m)).float()
        elif win_type == 'hann':
            return lambda m: torch.hann_window(m)

    def _wav_to_frame(self, x: torch.Tensor):
        """
        Given waveform audio, divide into (overlapping) frames.

        :param x: waveform audio of shape (n_batch, frame_len)
        :return: framed audio of shape (n_batch, n_frames, signal_length)
        """

        assert x.ndim == 2
        n_batch, signal_len = x.shape

        # compute required number of frames given stored frame length
        if self.win_type == 'rectangular':  # non-overlapping frames
            n_frames = signal_len // self.win_length + 1
            pad_len = n_frames * self.win_length
        elif self.win_type == 'hann':  # 50% overlap
            n_frames = signal_len // self.hop_length + 1
            pad_len = (n_frames - 1) * (self.win_length // 2) + self.win_length
        else:
            raise ValueError(f'Invalid window type {self.win_type}')

        # pad input audio to integer number of frames
        if signal_len < pad_len:
            x = F.pad(x, (0, pad_len - signal_len))
        else:
            x = x[..., :pad_len]

        # divide input audio into frames
        if self.win_type == 'rectangular':
            x = x.unfold(-1, self.win_length, self.win_length)
        else:
            x = x.unfold(-1, self.win_length, self.win_length // 2)

        return x

    def _frame_to_wav(self, x: torch.Tensor):
        """
        Given framed audio, resynthesize waveform via overlap-add.

        :param x: framed audio of shape (n_batch, n_frames, frame_len)
        :return: waveform audio of shape (n_batch, signal_length)
        """

        assert x.ndim == 3
        n_batch, n_frames, _ = x.shape

        # restore signal from frames using overlap-add
        if self.win_type == 'rectangular':

            pad_len = n_frames * self.win_length

            x = F.fold(
                x.permute(0, 2, 1),
                (1, pad_len),
                kernel_size=(1, self.win_length),
                stride=(1, self.win_length)
            ).reshape(n_batch, -1)

        else:

            pad_len = (n_frames - 1) * (self.win_length // 2) + self.win_length

            # obtain window function
            win = self._get_win_func(self.win_type)(
                self.win_length
            ).to(x).reshape(1, 1, -1)

            x = x * win  # apply windowing along final dimension

            # use `nn.functional.fold` to perform overlap-add synthesis; for
            # reference, see https://tinyurl.com/pw7mv9hh
            x = F.fold(
                x.permute(0, 2, 1),
                (1, pad_len),
                kernel_size=(1, self.win_length),
                stride=(1, self.win_length // 2)
            ).reshape(n_batch, -1)

        return x

    @staticmethod
    def _match_input(x: torch.Tensor, x_ref: torch.Tensor):
        """
        Given adversarial output, match scale and dimensions to original input

        :param x: adversarial audio of shape (n_batch, ..., adv_signal_length)
        :param x_ref: original audio of shape (n_batch, ..., signal_length)
        """

        assert x.ndim >= 2 and x_ref.ndim >= 2

        # match original dimensions
        if x.ndim < x_ref.ndim:
            x = x.unsqueeze(1)

        # prepare to peak-normalize output audio
        peak = torch.max(torch.abs(x_ref), -1)[0].reshape(-1)

        # peak-normalize to match input
        factor = peak / torch.max(torch.abs(x), -1)[0].reshape(-1)
        factor = factor.reshape(x.shape[0], *((1,)*(x.ndim - 1)))
        x = (x * factor)[..., :x_ref.shape[-1]]

        return x

    def _remove_frequencies(self, x: torch.Tensor):
        """
        Remove frequency content below relative energy threshold.

        :param x: framed audio of shape (n_batch, n_frames, frame_len)
        :return: perturbed audio frames, shape (n_batch, n_frames, frame_len)
        """

        # convert threshold to energy ratio
        threshold = 10 ** (-self.threshold_db / 10)

        assert x.ndim == 3  # (n_batch, n_frames, frame_len)
        n_batch = x.shape[0]

        # if frames overlap, pad input to ensure each sample occurs in the same
        # number of frames
        if self.win_type in ["hann", "triangular"]:
            x = F.pad(x, (self.win_length // 2, self.win_length // 2))

        # compute power spectral density (PSD) of each frame, doubling paired
        # frequencies (non-DC, non-nyquist)
        x_rfft = fft.rfft(x)
        x_psd = torch.abs(x_rfft) ** 2  # (n_batch, n_frames, n_fft)

        if x.shape[-1] % 2:  # odd: DC frequency
            x_psd[..., 1:] *= 2
        else:  # even: DC and Nyquist frequencies
            x_psd[..., 1:-1] *= 2

        # sort frequency bins ascending by PSD
        x_psd_index = torch.argsort(x_psd, dim=-1)
        reordered = torch.gather(x_psd, -1, x_psd_index)

        # compute cumulative frequency energy across bins
        cumulative = torch.cumsum(reordered, dim=-1)

        # set threshold relative to highest-energy bin
        norm_threshold = (threshold * cumulative[..., -1]).unsqueeze(-1)
        cutoff = torch.searchsorted(cumulative, norm_threshold, right=True)

        # zero bins below threshold, using sorted indices
        n_frames = x_rfft.shape[1]
        for i in range(n_batch):
            for j in range(n_frames):
                x_rfft[i, j, x_psd_index[i, j, :cutoff[i, j]]] = 0

        # invert to waveform audio
        x = fft.irfft(
            x_rfft,
            x.shape[-1]
        )

        # undo additional padding if necessary
        if self.win_type == "hann":
            x = x[..., self.win_length // 2: -self.win_length // 2]

        return x  # (n_batch, n_frames, frame_len)

    def set_reference(self, x: torch.Tensor):
        """
        Given reference input, initialize perturbation parameters accordingly
        and match input device.

        :param x: reference audio, shape (n_batch, n_channels, signal_length)
        """
        self.threshold_db = self.threshold_db.to(x.device)

    def set_threshold(self, threshold_db: float):
        self.threshold_db.fill_(threshold_db)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        """
        Apply perturbation to inputs.

        :param x: input audio, shape (n_batch, n_channels, signal_length)
        """

        # require batch dimension
        assert x.ndim >= 2
        n_batch, signal_len = x.shape[0], x.shape[-1]

        x_orig = x.clone().detach()

        # discard channel dimension
        if x.ndim > 2:
            x = x.mean(dim=1, keepdim=True).squeeze(1)

        x = self._wav_to_frame(x)
        x = self._remove_frequencies(x)
        x = self._frame_to_wav(x)
        x = self._match_input(x, x_orig)

        return x

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
