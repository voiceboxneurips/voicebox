import torch
import torch.nn as nn

from torchaudio.transforms import MelSpectrogram

from src.data import DataProperties

from typing import Union

################################################################################
# ResNetSE34V2 spectrogram convolutional model for speaker verification
################################################################################


class Resemblyzer(nn.Module):

    def __init__(self,
                 hidden_size: int = 256,
                 embedding_size: int = 256,
                 layers: int = 3,
                 win_length: int = 400,
                 hop_length: int = 160,
                 n_mels: int = 40,
                 **kwargs):
        """
        Resemblyzer speaker embedding model, based on the system proposed in Wan
        et al. 2020 (https://arxiv.org/pdf/1710.10467.pdf). Code adapted from
        https://github.com/resemble-ai/Resemblyzer.

        Parameters
        ----------

        Returns
        -------

        """
        super().__init__()

        # mel spectrogram
        self.spec = MelSpectrogram(
            sample_rate=DataProperties.get('sample_rate'),
            n_fft=win_length,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            center=True,
            window_fn=torch.hann_window,
            pad_mode='reflect',
            norm='slaney',
            mel_scale='slaney',
            power=2.0
        )

        # simple LSTM network for computing embeddings
        self.lstm = nn.LSTM(n_mels, hidden_size, layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, embedding_size)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        """
        Compute speaker embeddings for a batch of utterances.

        Parameters
        ----------
        x (Tensor):

        Returns
        -------
        emb (Tensor):

        """

        # require batch dimension
        assert x.ndim >= 2
        n_batch, *channel_dims, signal_len = x.shape

        # add channel dimension if necessary
        if len(channel_dims) == 0:
            x = x.unsqueeze(1)

        # discard channel dimensions
        x = x.mean(1)

        # compute mel spectrogram
        x = self.spec(x).permute(0, 2, 1)  # (n_batch, n_frames, n_mels)

        # extract embeddings from final hidden layer of network
        _, (hidden, _) = self.lstm(x)
        emb = self.relu(self.linear(hidden[-1]))  # (batch_size, embedding_size)

        return emb
