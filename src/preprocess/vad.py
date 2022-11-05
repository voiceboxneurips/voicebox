import torch
import math
import decimal

from typing import List

import torch.nn.functional as F

from torchaudio.transforms import MFCC

from src.simulation.component import Component

################################################################################
# Voice Activity Detection (VAD)
################################################################################


class KaldiStyleVAD(Component):
    """
    Kaldi-style Voice Activity Detection (VAD) module. Adapted from
    https://github.com/fsepteixeira/FoolHD/blob/main/code/utils/vad_cmvn.py
    """
    def __init__(self,
                 compute_grad: bool = True,
                 threshold: float = -15.0,
                 proportion_threshold: float = 0.12,
                 frame_len: float = 0.025,
                 hop_len: float = 0.010,
                 mean_scale: float = 0.5,
                 context: int = 2):
        super().__init__(compute_grad)

        self.threshold = threshold
        self.proportion_threshold = proportion_threshold
        self.mean_scale = mean_scale
        self.context = context
        self.diff_zero = mean_scale != 0
        self.unfold_size = 2 * context + 1
        self.frame_len = int(frame_len * self.sample_rate)
        self.hop_len = int(hop_len * self.sample_rate)

        # prepare to compute MFCC
        self.mfcc = MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=30,
            dct_type=2,
            norm='ortho',
            log_mels=True,
            melkwargs={
                'n_fft': self.frame_len,
                'hop_length': self.hop_len,
                'n_mels': 30,
                'f_min': 20,
                'f_max': self.sample_rate // 2,
                'power': 2.0,
                'center': True
            }
        )

    def forward(self, x: torch.Tensor):

        if x.shape[-1] < self.frame_len + self.hop_len:
            return x

        # require batch dimension
        assert x.ndim >= 2

        # require mono audio, discard channel dimension
        n_batch, slen = x.shape[0], x.shape[-1]
        x = x.reshape(n_batch, slen)

        # compute MFCC
        x_mfcc = self.mfcc(x).permute(0, 2, 1)  # (n_batch, n_frames, n_mfcc)

        # set device for energy threshold
        energy_threshold = torch.tensor([self.threshold]).to(x_mfcc.device)

        # first MFCC coefficient represents log energy
        log_energy = x_mfcc[:, :, 0]

        if self.diff_zero:
            energy_threshold = energy_threshold + self.mean_scale * log_energy.mean(dim=1)

        # prepare frame-wise mask
        mask = torch.ones_like(log_energy)

        # pad borders with symmetric context before striding
        mask = F.pad(mask, pad=(self.context, self.context), value=1.0)

        # get all (overlapping) context "windows"
        mask = mask.unfold(dimension=1, size=self.unfold_size, step=1)

        # number of values included in each context window
        den_count = mask.sum(dim=-1)

        # pad borders with symmetric context
        log_energy = F.pad(log_energy, pad=(self.context, self.context))

        # get all (overlapping) context "windows"
        log_energy = log_energy.unfold(
            dimension=1,
            size=self.unfold_size,
            step=1
        )

        # number of values in each context window above threshold
        num_count = log_energy.gt(
            energy_threshold.unsqueeze(-1).unsqueeze(-1)
        ).sum(dim=-1)

        # frame-by-frame mask
        mask = num_count.ge(den_count*self.proportion_threshold)

        # "fold" to obtain waveform mask
        mask_wav = mask.unsqueeze(-1).repeat_interleave(
            repeats=self.frame_len, dim=-1
        )
        mask_wav = torch.cat(
            [
                mask_wav[:, 0],
                mask_wav[:, 1:][:, :, self.frame_len - self.hop_len:].reshape(
                    n_batch, -1
                )
            ], dim=-1
        )
        left_trim = self.frame_len // 2
        right_trim = mask_wav.shape[-1] - left_trim - x.shape[-1]
        mask_wav = mask_wav[..., left_trim: -right_trim]

        # compute number of accepted samples per input waveform
        samples_per_row: List[int] = []
        for e in torch.sum(mask_wav, dim=-1):
            samples_per_row.append(e.item())

        # split resulting tensor to keep trimmed inputs separate
        split = torch.split(x[mask_wav], samples_per_row)

        # placeholder for outputs: (n_batch, 1, padded_length)
        final = torch.zeros_like(x).unsqueeze(1)  # pad to preserve length

        # concatenate and pad split views
        for i, tensor in enumerate(split):
            length = tensor.shape[-1]
            final[i, :, :length] = tensor

        return final[..., :slen]


class VAD(Component):
    """
    Apply Voice Activity Detection (VAD) while allowing for straight-through
    gradient estimation. For now, only supports simple energy-based method,
    and should be placed after normalization to avoid scale-dependence.
    """
    def __init__(self,
                 compute_grad: bool = True,
                 frame_len: float = 0.05,
                 threshold: float = -72
                 ):

        super().__init__(compute_grad)

        self.threshold = threshold
        self.frame_len = int(
            decimal.Decimal(
                frame_len * self.sample_rate
            ).quantize(
                decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP
            )
        )  # convert seconds to samples, round up

    def forward(self, x: torch.Tensor):

        # require batch dimension
        assert x.ndim >= 2

        # require mono audio, discard channel dimension
        n_batch, slen = x.shape[0], x.shape[-1]
        audio = x.reshape(n_batch, slen)

        eps = 1e-12  # numerical stability

        # determine number of frames
        if slen <= self.frame_len:
            n_frames = 1
        else:
            n_frames = 1 + int(
                math.ceil(
                    (1.0 * slen - self.frame_len) / self.frame_len)
            )

        # pad to integer frame length
        padlen = int(n_frames * self.frame_len)
        zeros = torch.zeros((x.shape[0], padlen - slen,)).to(x)
        padded = torch.cat((audio, zeros), dim=-1)

        # obtain strided (frame-wise) view of audio
        shape = (padded.shape[0], n_frames, self.frame_len)
        frames = torch.as_strided(
            padded,
            size=shape,
            stride=(padded.shape[-1], self.frame_len, 1)
        )

        # create frame-by-frame mask based on energy threshold
        mask = 20 * torch.log10(
            ((frames * self.scale).norm(dim=-1) / self.frame_len) + eps
        ) > self.threshold

        # turn frame-by-frame mask into sample-by-sample mask
        mask_wav = torch.repeat_interleave(mask, self.frame_len, dim=-1)
        samples_per_row = torch.sum(mask, dim=-1) * self.frame_len

        split = torch.split(padded[mask_wav], tuple(samples_per_row))

        # placeholder for outputs: (n_batch, 1, padded_length)
        final = torch.zeros_like(padded).unsqueeze(1)  # pad to preserve length

        # concatenate and pad split views
        for i, tensor in enumerate(split):
            length = tensor.shape[-1]
            final[i, :, :length] = tensor

        return final[..., :slen]

