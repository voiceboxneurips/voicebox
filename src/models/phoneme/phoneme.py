import torch
import torch.nn as nn
import torch.nn.functional as F

from torchaudio.transforms import MFCC

from typing import Callable

from src.data import DataProperties

################################################################################
# Phoneme predictor model from AC-VC (Ronssin & Cernak)
################################################################################


class Delta(nn.Module):
    """Causal delta computation"""
    def forward(self, x: torch.Tensor):

        x = F.pad(x, (0, 1))
        x = torch.diff(x, n=1, dim=-1)

        return x


class PPGEncoder(nn.Module):
    """
    Phonetic posteriorgram (PPG) predictor from Almost-Causal Voice Conversion
    """
    def __init__(self,
                 win_length: int = 256,
                 hop_length: int = 128,
                 win_func: Callable = torch.hann_window,
                 n_mels: int = 32,
                 n_mfcc: int = 13,
                 lstm_depth: int = 2,
                 hidden_size: int = 512
                 ):
        """
        Parameters
        ----------

        win_length (int):       spectrogram window length in samples

        hop_length (int):       spectrogram hop length in samples

        win_func (Callable):    spectrogram window function

        n_mels (int):           number of mel-frequency bins

        n_mfcc (int):           number of cepstral coefficients

        lstm_depth (int):       number of LSTM layers

        hidden_size (int):      hidden layer dimension for MLP and LSTM
        """

        super().__init__()

        self.win_length = win_length
        self.hop_length = hop_length

        # compute spectral representation
        mel_kwargs = {
            "n_fft": self.win_length,
            "win_length": self.win_length,
            "hop_length": self.hop_length,
            "window_fn": win_func,
            "n_mels": n_mels
        }
        self.mfcc = MFCC(
            sample_rate=DataProperties.get("sample_rate"),
            n_mfcc=n_mfcc,
            log_mels=True,
            melkwargs=mel_kwargs
        )

        # compute first- and second-order MFCC deltas
        self.delta = Delta()

        # PPG network
        self.mlp = nn.Sequential(
            nn.Linear(n_mfcc * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_depth,
            bias=True,
            batch_first=True,
            bidirectional=False
        )

    def forward(self, x: torch.Tensor):

        # require batch, channel dimensions
        assert x.ndim >= 2
        if x.ndim == 2:
            x = x.unsqueeze(1)

        # convert to mono audio
        x = x.mean(dim=1)

        mfcc = self.mfcc(x)  # (n_batch, n_mfcc, n_frames)
        delta1 = self.delta(mfcc)  # (n_batch, n_mfcc, n_frames)
        delta2 = self.delta(delta1)  # (n_batch, n_mfcc, n_frames)

        x = torch.cat([mfcc, delta1, delta2], dim=1)  # (n_batch, 3 * n_mfcc, n_frames)
        x = x.permute(0, 2, 1)  # (n_batch, n_frames, 3 * n_mfcc)

        x = self.mlp(x)  # (n_batch, n_frames, hidden_size)
        x, _ = self.lstm(x)  # (n_batch, n_frames, hidden_size)

        return x