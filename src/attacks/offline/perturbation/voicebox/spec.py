from src.attacks.offline.perturbation.voicebox import mlp
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torchaudio.transforms import Spectrogram, MelSpectrogram, MFCC

from src.attacks.offline.perturbation.voicebox.expnorm import ExponentialUnitNorm
from src.attacks.offline.perturbation.voicebox.batchnorm import BatchNorm
from src.attacks.offline.perturbation.voicebox.mlp import MLP
from src.data import DataProperties

################################################################################
# Convolutional spectrogram encoder with optional lookahead
################################################################################


class CausalPadding(nn.Module):
    """Perform 'causal' padding at end of signal along final tensor dimension"""

    def __init__(self, pad: int = 0):
        super().__init__()
        self.pad = pad

    def forward(self, x: torch.Tensor):
        return F.pad(x, (0, self.pad))


class SpectrogramEncoder(nn.Module):
    """Spectrogram encoder with optional lookahead"""
    def __init__(self,
                 win_length: int = 512,
                 win_type: str = 'hann',
                 spec_type: str = 'linear',
                 lookahead: int = 5,
                 hidden_size: int = 512,
                 n_mels: int = 64,
                 mlp_depth: int = 2,
                 normalize: str = None
                 ):
        super().__init__()

        # check validity of attributes
        assert normalize in [None, 'none', 'instance', 'exponential']
        if win_type not in ['rectangular', 'triangular', 'hann']:
            raise ValueError(f'Invalid window type {win_type}')

        # store attributes
        self.win_length = win_length
        self.win_type = win_type
        self.lookahead = lookahead
        self.hidden_size = hidden_size
        self.n_mels = n_mels
        self.spec_type = spec_type
        self.mlp_depth = mlp_depth
        self.normalize = normalize

        # determine hop length from window function
        if self.win_type == 'rectangular':  # non-overlapping frames
            self.hop_length = self.win_length
        else:
            self.hop_length = self.win_length // 2

        # determine spectrogram normalization method
        n_freq = n_mels if spec_type in ['mel', 'mfcc'] else win_length // 2 + 1

        if normalize in [None, 'none']:
            self.norm = nn.Identity()
        elif normalize == 'instance':
            self.norm = nn.InstanceNorm1d(
                num_features=n_freq,
                track_running_stats=True
            )
        elif normalize == 'exponential':
            self.norm = ExponentialUnitNorm(
                decay=1.0,
                hop_size=self.hop_length,
                n_freq=n_freq
            )

        # compute spectral representation
        spec_kwargs = {
            "n_fft": self.win_length,
            "win_length": self.win_length,
            "hop_length": self.hop_length,
            "window_fn": self._get_win_func(self.win_type),
        }
        mel_kwargs = {**spec_kwargs, "n_mels": self.n_mels}

        if spec_type == 'linear':
            self.spec = Spectrogram(
                **spec_kwargs
            )
        elif spec_type == 'mel':
            self.spec = MelSpectrogram(
                sample_rate=DataProperties.get("sample_rate"),
                **mel_kwargs
            )
        elif spec_type == 'mfcc':
            self.spec = MFCC(
                sample_rate=DataProperties.get("sample_rate"),
                n_mfcc=self.n_mels,
                log_mels=True,
                melkwargs=mel_kwargs
            )

        # GLU - learn which channels of input to pass through most strongly
        self.glu = nn.Sequential(
            nn.Conv1d(
                in_channels=n_freq,
                out_channels=self.hidden_size * 2,
                kernel_size=1,
                stride=1),
            nn.GLU(dim=1)
        )

        # Conv1D layers
        conv = []
        for i in range(lookahead):
            conv.extend([
                CausalPadding(1),
                nn.Conv1d(
                    in_channels=self.hidden_size,
                    out_channels=self.hidden_size,
                    kernel_size=2,
                    stride=1
                ),
                BatchNorm(num_features=self.hidden_size, feature_dim=1) if i < lookahead - 1 else nn.Identity(),
                nn.ReLU()
            ])
        self.conv = nn.Sequential(*conv)

        # pre-bottleneck MLP
        self.mlp = MLP(
            in_channels=self.hidden_size,
            hidden_size=self.hidden_size,
            depth=mlp_depth
        )

    @staticmethod
    def _get_win_func(win_type: str):
        if win_type == 'rectangular':
            return lambda m: torch.ones(m)
        elif win_type == 'hann':
            return lambda m: torch.hann_window(m)
        elif win_type == 'triangular':
            return lambda m: torch.as_tensor(np.bartlett(m)).float()

    def forward(self, x: torch.Tensor):

        # ensure the input contains at least a single frame of audio
        assert x.shape[-1] >= self.win_length

        # require batch, channel dimensions
        assert x.ndim >= 2
        if x.ndim == 2:
            x = x.unsqueeze(1)

        # convert to mono audio
        x = x.mean(dim=1)

        # compute spectrogram
        spec = self.spec(x) + 1e-6  # (n_batch, n_freq, n_frames)

        if self.spec_type in ['linear', 'mel']:
            spec = 10 * torch.log10(spec + 1e-8)  # (n_batch, n_freq, n_frames)

        # normalize spectrogram
        spec = self.norm(spec)  # (n_batch, n_freq, n_frames)

        # actual encoder network
        encoded = self.glu(spec)  # (n_batch, hidden_size, n_frames)
        encoded = self.conv(encoded)  # (n_batch, hidden_size, n_frames)
        encoded = self.mlp(
            encoded.permute(0, 2, 1)
        )  # (n_batch, n_frames, hidden_size)

        return encoded
