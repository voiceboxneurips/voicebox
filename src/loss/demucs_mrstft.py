import torch
import torch.nn.functional as F

from typing import Collection

from src.loss.auxiliary import AuxiliaryLoss

################################################################################
# Demucs-style multi-resolution STFT loss
################################################################################


class DemucsMRSTFTLoss(AuxiliaryLoss):
    """
    Compute multi-resolution spectrogram loss, as proposed by Yamamoto et al.
    (https://arxiv.org/abs/1910.11480). Uses linear and log-scaled spectrograms
    with spectral convergence loss, as in Defossez et al.
    (https://arxiv.org/abs/2006.12847). Code adapted from
    https://github.com/facebookresearch/denoiser.
    """
    def __init__(self,
                 reduction: str = 'none',
                 fft_sizes: Collection = (1024, 2048, 512),
                 hop_sizes: Collection = (120, 240, 50),
                 win_lengths: Collection = (600, 1200, 240),
                 window: str = 'hann',
                 factor_sc: float = 0.1,
                 factor_mag: float = 0.1):
        super().__init__(reduction)

        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)

        # store window functions as buffers
        for i, win_length in enumerate(win_lengths):
            self.register_buffer(
                f'window_{i}',
                self._get_win_func(window)(win_length)
            )

        # store STFT parameters at each resolution
        self.stft_params = list(
            zip(
                fft_sizes,
                hop_sizes,
                win_lengths
            )
        )

        # scale losses
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag

        # prepare to store reference spectrograms
        self.ref_mag = None

    @staticmethod
    def _get_win_func(win_type: str):
        if win_type == 'rectangular':
            return lambda m: torch.ones(m)
        elif win_type == 'hann':
            return lambda m: torch.hann_window(m)
        elif win_type == 'hamming':
            return lambda m: torch.hamming_window(m)
        elif win_type == 'kaiser':
            return lambda m: torch.kaiser_window(m)
        else:
            raise ValueError(f'Invalid window function {win_type}')

    @staticmethod
    def _pad(x: torch.Tensor, win_length: int, hop_length: int):
        """
        Avoid boundary artifacts by padding inputs before STFT such that all
        samples are represented in the same number of spectrogram windows
        """
        pad_frames = win_length // hop_length - 1
        pad_len = pad_frames * hop_length
        return F.pad(x, (pad_len, pad_len))

    def _stft(self,
              x: torch.Tensor,
              fft_size: int,
              hop_size: int,
              win_length: int,
              window: torch.Tensor) -> torch.Tensor:
        """
        Perform STFT and convert to magnitude spectrogram.
        :param x: waveform audio; shape (n_batch, n_channels, signal_length)
        :param fft_size: FFT size in samples
        :param hop_size: hop size in samples
        :param win_length: window length in samples
        :param window: window function
        :return: tensor holding magnitude spectrogram; shape
                 (n_batch, n_channels, n_frames, fft_size // 2 + 1)
        """

        # require batch dimension
        assert x.ndim >= 2
        if x.ndim == 2:
            x = x.unsqueeze(1)

        # pad to avoid boundary artifacts
        x = self._pad(x, win_length, hop_size)

        # reshape to handle multi-channel audio
        n_batch, n_channels, signal_length = x.shape
        x = x.view(n_batch * n_channels, signal_length)

        # compute STFT
        x_stft = torch.stft(
            x,
            fft_size,
            hop_size,
            win_length,
            window
        )
        _, n_freq, n_frames, _ = x_stft.shape
        mag_stft = x_stft[..., 0] ** 2 + x_stft[..., 1] ** 2

        return torch.sqrt(
            torch.clamp(
                mag_stft, min=1e-7)
        ).transpose(-2, -1).view(n_batch, n_channels, n_frames, n_freq)

    @staticmethod
    def _spectral_convergence(x_mag: torch.Tensor,
                              x_ref_mag: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral convergence loss between magnitude spectrograms.

        :param x_mag: magnitude spectrogram, shape
                      (n_batch, n_channels, n_frames, n_freq)
        :param x_ref_mag: reference magnitude spectrogram, shape
        :return: unreduced batch loss of shape (n_batch, n_channels)
        """

        # require batch dimension
        assert x_mag.ndim >= 3

        # flatten spectrogram dimensions
        x_mag = x_mag.reshape(x_mag.shape[0], 1, -1)
        x_ref_mag = x_ref_mag.reshape(x_ref_mag.shape[0], 1, -1)

        # numerical stability; otherwise, can end up with NaN gradients
        # when x_mag == x_ref_mag
        eps = 1e-12

        return torch.linalg.matrix_norm(
            x_ref_mag - x_mag + eps, ord="fro"
        ) / torch.linalg.matrix_norm(
            x_ref_mag, ord="fro"
        )

    @staticmethod
    def _log_magnitude(x_mag: torch.Tensor,
                       x_ref_mag: torch.Tensor) -> torch.Tensor:
        """
        Compute log loss between magnitude spectrograms.

        :param x_mag: magnitude spectrogram, shape
                      (n_batch, n_channels, n_frames, n_freq)
        :param x_ref_mag: reference magnitude spectrogram, shape
        :return: unreduced batch loss of shape (n_batch, n_channels)
        """

        assert x_mag.ndim >= 3
        n_batch = x_mag.shape[0]

        return torch.mean(
            torch.abs(
                torch.log(
                    x_mag
                ).reshape(n_batch, -1) - torch.log(
                    x_ref_mag
                ).reshape(n_batch, -1)
            ),
            dim=-1
        )

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

        # require batch dimension
        assert x.ndim >= 2

        if x_ref is not None:
            assert x_ref.ndim >= 2

        # compute magnitude and spectral convergence losses
        sc_loss = torch.zeros(x.shape[0]).to(x.device)
        mag_loss = torch.zeros(x.shape[0]).to(x.device)

        for i, stft_params in enumerate(self.stft_params):

            window = list(self.buffers())[i]

            # compute input magnitude spectrogram
            x_mag = self._stft(x, *stft_params, window)

            # compute or load reference magnitude spectrogram
            if x_ref is not None:
                x_ref_mag = self._stft(x_ref, *stft_params, window)
            else:
                x_ref_mag = self.ref_mag[i]

            # check compatibility of input and reference spectrograms
            assert self._check_broadcastable(
                x_mag, x_ref_mag
            ), f"Cannot broadcast inputs of shape {x_mag.shape} " \
               f"with reference of shape {x_ref_mag.shape}"

            sc_loss += self._spectral_convergence(x_mag, x_ref_mag)
            mag_loss += self._log_magnitude(x_mag, x_ref_mag)

        sc_loss /= len(self.stft_params)
        mag_loss /= len(self.stft_params)

        return sc_loss * self.factor_sc + mag_loss * self.factor_mag

    def set_reference(self, x_ref: torch.Tensor):

        # require batch dimension, discard channel dimension
        assert x_ref.ndim >= 2

        # store reference spectrogram for each scale
        self.ref_mag = []

        for i, stft_params in enumerate(self.stft_params):

            window = list(self.buffers())[i]

            x_ref_mag = self._stft(x_ref, *stft_params, window)
            self.ref_mag.append(x_ref_mag)
