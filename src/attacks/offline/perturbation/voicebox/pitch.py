import torch
import torch.nn as nn
import torch.nn.functional as F

import torchcrepe
import pyworld

import numpy as np

from src.data import DataProperties

################################################################################
# Compute frame-wise pitch and periodicity estimates
################################################################################


class PitchEncoder(nn.Module):

    def __init__(self,
                 algorithm: str = 'dio',
                 return_periodicity: bool = True,
                 hop_length: int = 128,
                 ):

        super().__init__()

        self.algorithm = algorithm
        self.return_periodicity = return_periodicity
        self.hop_length = hop_length

    @torch.no_grad()
    def forward(self, x: torch.Tensor):

        # ensure the input contains at least a single frame of audio
        assert x.shape[-1] >= self.hop_length

        # avoid modifying input audio
        n_batch, *channel_dims, signal_len = x.shape

        # add channel dimension if necessary
        if len(channel_dims) == 0:
            x = x.unsqueeze(1)

        x = x.mean(dim=1)

        if self.algorithm == 'torchcrepe':

            if not self.return_periodicity:
                pitch = torchcrepe.predict(
                    x,
                    DataProperties.get('sample_rate'),
                    hop_length=self.hop_length,
                    fmin=50,
                    fmax=550,
                    model='tiny',
                    batch_size=10_000,
                    return_periodicity=False,
                    device='cuda',
                )

                # (n_batch, n_frames, 1)
                return pitch.unsqueeze(-1)
            else:
                pitch, periodicity = torchcrepe.predict(
                    x,
                    DataProperties.get('sample_rate'),
                    hop_length=self.hop_length,
                    fmin=50,
                    fmax=550,
                    model='tiny',
                    batch_size=10_000,
                    return_periodicity=True,
                    device='cuda',
                )

                # (n_batch, n_frames, 1), (n_batch, n_frames, 1)
                return pitch.unsqueeze(-1), periodicity.unsqueeze(-1)

        elif self.algorithm == 'dio':

            pitch_out, periodicity_out, device = [], [], x.device
            hop_ms = 1000*self.hop_length/DataProperties.get('sample_rate')
            x_np = x.clone().double().cpu().numpy()

            for i in range(n_batch):
                pitch, timeaxis = pyworld.dio(
                    x_np[i],
                    fs=DataProperties.get('sample_rate'),
                    f0_floor=50,
                    f0_ceil=550,
                    frame_period=hop_ms,
                    speed=4)  # downsampling factor, for speedup
                pitch = pyworld.stonemask(
                    x_np[i],
                    pitch,
                    timeaxis,
                    DataProperties.get('sample_rate'))

                pitch_out.append(pitch)

                if self.return_periodicity:
                    unvoiced = pyworld.d4c(
                        x_np[i],
                        pitch,
                        timeaxis,
                        DataProperties.get('sample_rate'),
                    ).mean(axis=1)

                    periodicity_out.append(unvoiced)

            pitch_out = torch.as_tensor(
                np.stack(pitch_out, axis=0),
                dtype=torch.float32,
                device=device).unsqueeze(-1)

            if not self.return_periodicity:
                # (n_batch, n_frames, 1)
                return pitch_out
            else:
                periodicity_out = torch.as_tensor(
                    np.stack(periodicity_out, axis=0),
                    dtype=torch.float32,
                    device=device).unsqueeze(-1)  # remove unsqueeze if not averaging!

                # (n_batch, n_frames), (n_batch, n_frames, 1)
                return pitch_out, periodicity_out

        else:
            raise ValueError(f'Invalid algorithm {self.algorithm}')
