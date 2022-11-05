import torch
import torch.nn as nn
import torch.nn.functional as F

from torchaudio.transforms import MFCC

from src.loss.auxiliary import AuxiliaryLoss
from src.data import DataProperties

################################################################################
# MFCC cosine loss; measures scale-independent spectral distance
################################################################################


class MFCCCosineLoss(AuxiliaryLoss):
    """
    Compute frame-wise cosine distance between MFCC representations of an input
    and reference. This serves as a scale-independent spectral distance.
    """
    def __init__(self,
                 reduction: str = 'none',
                 n_mfcc: int = 30,
                 log_mels: bool = True,
                 n_mels: int = 30,
                 win_length: float = 0.025,
                 hop_length: float = 0.010):
        super().__init__(reduction)

        self.sample_rate = DataProperties.get("sample_rate")

        self.win_length = int(win_length * self.sample_rate)
        self.hop_length = int(hop_length * self.sample_rate)

        self.mfcc = MFCC(n_mfcc=n_mfcc,
                         sample_rate=self.sample_rate,
                         norm='ortho',
                         log_mels=log_mels,
                         melkwargs={
                             'n_mels': n_mels,
                             'n_fft': self.win_length,
                             'win_length': self.win_length,
                             'hop_length': self.hop_length,
                             'f_min': 20.0,
                             'f_max': self.sample_rate // 2,
                             'window_fn': torch.hann_window}
                         )
        self.cos = nn.CosineSimilarity(dim=-2)  # compute per frame
        self.ref_mfcc = None

    def _pad(self, x: torch.Tensor):
        """
        Avoid boundary artifacts by padding inputs before STFT such that all
        samples are represented in the same number of spectrogram windows
        """
        pad_frames = self.win_length // self.hop_length - 1
        pad_len = pad_frames * self.hop_length
        return F.pad(x, (pad_len, pad_len))

    def _compute_loss(self, x: torch.Tensor, x_ref: torch.Tensor = None):
        """
        Compute frame-wise cosine similarity between MFCC representations of
        input and reference. If no reference is provided, a stored reference
        will be used.

        :param x: input, shape (n_batch, n_channels, signal_length)
        :param x_ref: reference, shape (n_batch, n_channels, signal_length) or
                      (1, n_channels, signal_length)
        :return: loss, shape (n_batch,)
        """

        # require batch dimension, discard channel dimension
        assert x.ndim >= 2
        n_batch, signal_length = x.shape[0], x.shape[-1]
        x = x.reshape(n_batch, signal_length)

        if x_ref is not None:
            assert x_ref.ndim >= 2
            n_batch, signal_length = x_ref.shape[0], x_ref.shape[-1]
            x_ref = x_ref.reshape(n_batch, signal_length)

        # compute input MFCC representation
        x_pad = self._pad(x)
        x_mfcc = self.mfcc(x_pad)  # (n_batch, n_mfcc, n_frames)

        # compute reference MFCC representation
        if x_ref is None:
            x_ref_mfcc = self.ref_mfcc
        else:
            x_ref_pad = self._pad(x_ref)
            x_ref_mfcc = self.mfcc(x_ref_pad)

        # ensure broadcastable inputs
        assert self._check_broadcastable(
            x_mfcc, x_ref_mfcc
        ), f"Cannot broadcast inputs of shape {x_mfcc.shape} " \
           f"with reference of shape {x_ref_mfcc.shape}"

        cos_dist = 1-self.cos(x_mfcc, x_ref_mfcc)

        # taking mean along time dimension, rather than sum, prevents signal
        # length from influencing loss magnitude
        return cos_dist.mean(dim=-1)

    def set_reference(self, x_ref: torch.Tensor):

        # require batch dimension, discard channel dimension
        assert x_ref.ndim >= 2
        n_batch, signal_length = x_ref.shape[0], x_ref.shape[-1]
        x_ref = x_ref.reshape(n_batch, signal_length)

        self.mfcc.to(x_ref.device)  # adopt reference device

        # pad to avoid boundary artifacts
        x_ref_pad = self._pad(x_ref).clone().detach()
        self.ref_mfcc = self.mfcc(x_ref_pad)
