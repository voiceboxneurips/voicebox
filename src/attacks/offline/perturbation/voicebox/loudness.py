import torch
import torch.nn as nn

import librosa as li
import numpy as np

from src.data import DataProperties

################################################################################
# Extract frame-wise A-weighted loudness
################################################################################


class LoudnessEncoder(nn.Module):
    """Extract frame-wise A-weighted loudness"""
    def __init__(self,
                 hop_length: int = 128,
                 n_fft: int = 256
                 ):

        super().__init__()

        self.hop_length = hop_length
        self.n_fft = n_fft

    def forward(self, x: torch.Tensor):

        n_batch, *channel_dims, signal_len = x.shape

        # require batch, channel dimensions
        assert x.ndim >= 2
        if x.ndim == 2:
            x = x.unsqueeze(1)

        # convert to mono audio
        x = x.mean(dim=1)

        spec = li.stft(
            x.detach().cpu().numpy(),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            center=True,
        )
        spec = np.log(abs(spec) + 1e-7)

        # compute A-weighting curve for frequencies of analysis
        f = li.fft_frequencies(
            sr=DataProperties.get('sample_rate'), n_fft=self.n_fft)
        a_weight = li.A_weighting(f)

        # apply multiplicative weighting via addition in log domain
        spec = spec + a_weight.reshape(1, -1, 1)

        # take mean over each frame
        loudness = torch.from_numpy(np.mean(spec, 1)).unsqueeze(-1).float().to(x.device)

        return loudness
