import torch
import torch.nn.functional as F

from typing import Iterable

from src.loss.auxiliary import AuxiliaryLoss

################################################################################
# DDSP-style multi-resolution STFT loss
################################################################################


class MRSTFTLoss(AuxiliaryLoss):
    """
    Compute multi-resolution spectrogram loss, as proposed by Yamamoto et al.
    (https://arxiv.org/abs/1910.11480). Uses linear and log-scaled spectrograms,
    as in Engel et al. (https://arxiv.org/abs/2001.04643).
    """
    def __init__(self,
                 reduction: str = 'none',
                 scales: Iterable[int] = (4096, 2048, 1024, 512, 256, 128),
                 overlap: float = 0.75):
        super().__init__(reduction)

        self.scales = scales
        self.overlap = overlap

        self.ref_wav = None
        self.ref_stft = None

    @staticmethod
    def _safe_log(x: torch.Tensor):
        return torch.log(x + 1e-7)

    @staticmethod
    def _pad(x: torch.Tensor, win_length: int, hop_length: int):
        """
        Avoid boundary artifacts by padding inputs before STFT such that all
        samples are represented in the same number of spectrogram windows
        """
        pad_frames = win_length // hop_length - 1
        pad_len = pad_frames * hop_length
        return F.pad(x, (pad_len, pad_len))

    def _stft(self, x: torch.Tensor, scale: int):

        # pad input to avoid boundary artifacts
        x_pad = self._pad(
            x,
            win_length=scale,
            hop_length=int(scale * (1 - self.overlap))
        )

        # compute STFT at given window length
        return torch.stft(
            x_pad,
            n_fft=scale,
            hop_length=int(scale * (1 - self.overlap)),
            win_length=scale,
            window=torch.hann_window(scale).to(x_pad.device),
            center=True,
            normalized=True,
            return_complex=True
        ).abs()

    def set_reference(self, x_ref: torch.Tensor):

        # require batch dimension, discard channel dimension
        assert x_ref.ndim >= 2
        n_batch, signal_length = x_ref.shape[0], x_ref.shape[-1]
        x_ref = x_ref.reshape(n_batch, signal_length)

        # store reference spectrogram for each scale
        self.ref_stft = []

        for scale in self.scales:
            x_ref_stft = self._stft(x_ref, scale)
            self.ref_stft.append(x_ref_stft)

    def _compute_loss(self, x: torch.Tensor, x_ref: torch.Tensor = None):
        """
        Compute multi-resolution spectrogram loss between input and reference,
        using both linear and log-scaled spectrograms of varying window lengths.
        If no reference is provided, a stored reference will be used.

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

        # compute loss on linear and log-scaled spectrograms
        lin_loss = torch.zeros(x.shape[0]).to(x.device)
        log_loss = torch.zeros(x.shape[0]).to(x.device)

        for i, scale in enumerate(self.scales):

            x_stft = self._stft(x, scale)

            if x_ref is not None:
                x_ref_stft = self._stft(x_ref, scale)
            else:
                x_ref_stft = self.ref_stft[i]

            # check compatibility of input and reference spectrograms
            assert self._check_broadcastable(
                x_stft, x_ref_stft
            ), f"Cannot broadcast inputs of shape {x_stft.shape} " \
               f"with reference of shape {x_ref_stft.shape}"

            lin_loss += (x_stft - x_ref_stft).abs().mean()
            log_loss += (
                    self._safe_log(x_stft) - self._safe_log(x_ref_stft)
            ).abs().mean()

        return lin_loss + log_loss
