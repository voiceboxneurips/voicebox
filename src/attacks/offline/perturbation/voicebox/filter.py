import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft

import numpy as np

from typing import Union
import warnings

################################################################################
# Time-varying FIR filter module
################################################################################


class FilterLayer(nn.Module):
    """Encapsulate FIR filtering in network layer"""
    def __init__(self,
                 win_length: int = 512,
                 win_type: str = 'hann',
                 n_bands: int = 128,
                 normalize_ir: Union[str, int, float] = None,
                 **kwargs):
        """
        Given a set of frame-wise controls specifying the frequency amplitude
        response of a time-varying FIR filter, apply filter to incoming audio.

        Parameters
        ----------
        win_length (int):   analysis window length in samples

        win_type (str):     window function; must be one of 'rectangular',
                            'triangular', or 'hann'

        n_bands (int):      number of filter bands

        normalize_ir (int): type of normalization to apply to FIR impulse
                            responses; must be 1, 2, or None
        """
        super().__init__()

        if win_type not in ['rectangular', 'triangular', 'hann']:
            raise ValueError(f'Invalid window type {win_type}')

        if normalize_ir not in [None, 'none', 1, 2]:
            raise ValueError(
                f'Invalid IR normalization type {normalize_ir}')

        # round window size to next power of 2
        next_pow_2 = 2**(win_length - 1).bit_length()
        if win_length != next_pow_2:
            warnings.warn(f'Rounding block size {win_length} to nearest power'
                          f' of 2 ({next_pow_2})')

        # store attributes
        self.win_length = next_pow_2
        self.n_bands = n_bands
        self.win_type = win_type
        self.normalize_ir = normalize_ir

        # determine hop length from window function
        if self.win_type == 'rectangular':  # non-overlapping frames
            self.hop_length = self.win_length
        else:  # overlapping frames
            self.hop_length = self.win_length // 2

    @staticmethod
    def _get_win_func(win_type: str):
        """Obtain callable window function by name"""
        if win_type == 'rectangular':
            return lambda m: torch.ones(m)
        elif win_type == 'hann':
            return lambda m: torch.hann_window(m)
        elif win_type == 'triangular':
            return lambda m: torch.as_tensor(np.bartlett(m)).float()

    def _amp_to_ir(self, amp: torch.Tensor):
        """
        Convert filter frequency amplitude response into a time-domain impulse
        response. The filter response is given as a per-frame transfer function,
        and a symmetric impulse response is returned.

        Parameters
        ----------
        amp (torch.Tensor): shape (n_batch, n_frames, n_bands) or
                            (1, n_frames, n_bands); holds per-frame frequency
                            magnitude response of time-varying FIR filter

        Returns
        -------
        impulse (torch.Tensor): shape (n_batch, n_frames, 2 * n_bands + 1)
        """

        # convert to complex zero-phase representation
        amp = torch.stack([amp, torch.zeros_like(amp)], -1)
        amp = torch.view_as_complex(amp)  # (n_batch, n_frames, n_bands)

        # compute 1D inverse FFT along final dimension, treating bands as
        # Fourier frequencies of analysis
        impulse = fft.irfft(amp)

        # require filter size to match time-domain transform of filter bands
        filter_size = impulse.shape[-1]

        # apply window to shifted zero-phase (symmetric) form of impulse
        impulse = torch.roll(impulse, filter_size // 2, -1)
        win = torch.hann_window(
            filter_size, dtype=impulse.dtype, device=impulse.device
        )

        if self.normalize_ir is None: # or self.normalize_ir ==  'none': disabled string option for jit.scripting. 
            pass
        elif self.normalize_ir == 1:
            impulse = impulse / (torch.sum(
                impulse, dim=-1, keepdim=True
            ) + 1e-20)
        elif self.normalize_ir == 2:
            impulse = impulse / torch.norm(
                impulse, p=2, dim=-1, keepdim=True
            ) + 1e-20

        return impulse * win

    def _fft_convolve(self,
                      signal: torch.Tensor,
                      kernel: torch.Tensor,
                      n_fft: int):
        """
        Given waveform representations of signal and FIR filter, convolve
        via point-wise multiplication in Fourier domain

        Parameters
        ----------
        signal (torch.Tensor): shape (n_batch, n_frames, win_length); holds
                               framed audio input

        kernel (torch.Tensor): shape (n_batch, n_frames, 2 * n_bands + 1); holds
                               time-domain impulse responses corresponding to
                               filter controls for each frame

        n_fft (int):           number of FFT bins

        Returns
        -------
        convolved (torch.Tensor): shape (n_batch, n_frames, n_fft); holds
                                  filtered audio
        """

        # right-pad kernel and frames to n_fft samples
        signal = nn.functional.pad(signal, (0, n_fft - signal.shape[-1]))
        kernel = nn.functional.pad(kernel, (0, n_fft - kernel.shape[-1]))

        # apply convolution in Fourier domain and invert
        convolved = fft.irfft(fft.rfft(signal) * fft.rfft(kernel))

        # account for frame-by-frame delay
        rolled = torch.roll(convolved, shifts=-(self.n_bands - 1), dims=-1)

        return rolled

    def _pad_and_frame(self, x: torch.Tensor, n_frames: int):
        """
        Pad audio to given frame length and divide into frames

        Parameters
        ----------
        x (torch.Tensor): shape (n_batch, n_channels, signal_len); input audio

        n_frames (int):   target length in frames

        Returns
        -------
        framed (torch.Tensor): shape (n_batch, 1, n_frames, win_length); holds
                               padded and framed input audio
        """
        n_batch, *channel_dims, signal_len = x.shape

        if self.win_type == 'rectangular':
            pad_len = n_frames * self.win_length
        elif self.win_type in ['triangular', 'hann']:
            pad_len = (n_frames - 1) * (self.win_length // 2) + self.win_length
        else:
            raise ValueError(f'Invalid window type {self.win_type}')

        # apply padding/trim
        if signal_len < pad_len:
            x = nn.functional.pad(x, (0, pad_len - x.shape[-1]))
        else:
            x = x[..., :pad_len]

        # divide audio into frames
        x = x.unfold(-1, self.win_length, self.hop_length)

        return x

    def _ola(self, x: torch.Tensor):
        """
        Re-synthesize waveform audio from filtered frames via overlap-add (OLA)

        Parameters
        ----------
        x (torch.Tensor): shape (n_batch, n_frames, n_fft); holds framed and
                          filtered audio

        n_fft (int):      length of FFT

        Returns
        -------
        synthesized (torch.Tensor): shape (n_batch, n_channels, signal_len);
                                    holds reconstructed audio
        """

        # check dimensions
        assert x.ndim == 3
        n_batch, n_frames, n_fft = x.shape

        if self.win_type == 'rectangular':

            # compute target output length
            pad_len = self.win_length * (n_frames - 1) + n_fft

            x = nn.functional.fold(
                x.permute(0, 2, 1),
                (1, pad_len),
                kernel_size=(1, n_fft),
                stride=(1, self.win_length)
            ).reshape(n_batch, -1)

        elif self.win_type in ['triangular', 'hann']:

            # compute target output length
            pad_len = (n_frames - 1) * (self.win_length // 2) + self.win_length
            truncated_len = ((pad_len - self.win_length)
                             // (self.win_length // 2)
                             ) * (self.win_length // 2) + n_fft

            # obtain window functions and pad to match frame length
            win = self._get_win_func(
                self.win_type
            )(self.win_length).to(x).reshape(1, 1, -1)
            win_pad_len = x.shape[-1] - win.shape[-1]
            win = nn.functional.pad(win, (0, win_pad_len))

            # apply window frame-by-frame
            x = x * win

            # use `nn.functional.fold` to perform overlap-add synthesis; for
            # reference, see https://tinyurl.com/pw7mv9hh
            x = nn.functional.fold(
                x.permute(0, 2, 1),
                (1, truncated_len),
                kernel_size=(1, n_fft),
                stride=(1, self.win_length // 2)
            ).reshape(n_batch, -1)

        return x

    def forward(self, x: torch.Tensor, controls: torch.Tensor):
        """
        Use given controls to parameterize a time-varying FIR filter and apply
        to incoming audio

        Parameters
        ----------
        x (torch.Tensor):        shape (n_batch, n_channels, signal_len); holds
                                 the input audio
        controls (torch.Tensor): shape (n_batch, n_frames, n_bands); holds
                                 frame-wise filter controls
        """

        # ensure the input contains at least a single frame of audio
        assert x.shape[-1] >= self.win_length

        # check filter control dimensions
        assert controls.shape[-1] == self.n_bands

        # avoid modifying input audio
        x = x.clone().detach()
        n_batch, *channel_dims, signal_len = x.shape

        # require batch, channel dimensions
        assert x.ndim >= 2
        if x.ndim == 2:
            x = x.unsqueeze(1)

        # convert to mono audio
        x = x.mean(dim=1)

        # pad or trim signal to match number of frames in filter controls
        n_frames = controls.shape[1]
        x = self._pad_and_frame(x, n_frames)

        # convert filter controls (frequency amplitude responses) to frame-by-
        # frame time-domain impulse responses
        impulse = self._amp_to_ir(controls)

        # determine FFT size using inferred FIR waveform filter length
        # (accounting for padding)
        n_fft_min = self.win_length + 2 * (self.n_bands - 1)
        n_fft = pow(2, math.ceil(math.log2(n_fft_min)))  # use next power of 2

        # convolve frame-by-frame in FFT domain; resulting padded frames will
        # contain "ringing" overlapping segments which must be summed
        x = self._fft_convolve(
            x,
            impulse,
            n_fft
        ).contiguous()  # (n_batch, n_frames_overlap, n_fft)

        # restore signal from frames using overlap-add
        x = self._ola(x)

        # match original dimensions
        x = x[..., :signal_len].reshape(
            n_batch,
            *((1,) * len(channel_dims)),
            signal_len
        )  # trim signal to original length

        return x
