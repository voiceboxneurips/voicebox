import random
import torch

import torch.fft as fft
import torch.nn.functional as F
import librosa as li

from pathlib import Path

from src.simulation.effect import Effect

################################################################################
# Convolutional reverb effect
################################################################################


class Reverb(Effect):
    """
    Apply impulse responses sampled from a given dataset
    """
    def __init__(self,
                 compute_grad: bool = True,
                 rir_dir: str = None,
                 ext: str = "wav",
                 fft_convolve: bool = True
                 ):
        super().__init__(compute_grad)
        self.rir_dir = rir_dir
        self.ext = ext
        self.fft_convolve = fft_convolve

        self.rir_list = list(Path(self.rir_dir).rglob(f'*.{self.ext}'))

        # store room impulse response as buffer to allow device movement
        self.register_buffer("rir", torch.zeros(1, dtype=torch.float32))

        # initialize RIR
        self.sample_params()

    @staticmethod
    def _fft_convolve(signal: torch.Tensor, kernel: torch.Tensor):

        # ensure signal and kernel have channel dimension
        signal = signal.reshape(signal.shape[0], -1)
        kernel = kernel.reshape(kernel.shape[0], -1)

        signal_len = signal.shape[-1]
        kernel_len = kernel.shape[-1]
        kernel = F.pad(
            kernel, (0, signal_len - kernel_len)
        )

        signal = F.pad(signal, (0, signal.shape[-1]))
        kernel = F.pad(kernel, (kernel.shape[-1], 0))

        output = fft.irfft(fft.rfft(signal) * fft.rfft(kernel))
        output = output[..., output.shape[-1] // 2:]

        return output.unsqueeze(1)

    def forward(self, x: torch.Tensor):

        n_batch = x.shape[0]
        if len(x.shape) < 3:
            x = x.reshape(n_batch, 1, -1)

        if self.fft_convolve:
            return self._fft_convolve(x, self.rir.clone().to(x))
        else:
            pad = F.pad(x, (self.rir.shape[-1]-1, 0))
            return F.conv1d(pad, self.rir.clone().to(x))

    def sample_params(self):
        """
        Sample and preprocess room impulse response
        """

        rir_np, _ = li.load(random.choice(self.rir_list),
                            sr=self.sample_rate, mono=True)
        rir = torch.as_tensor(rir_np)

        # trim leading silence
        offsets = torch.where(torch.abs(rir) > (torch.abs(rir).max() / 100))[0]
        rir = rir[..., offsets[0]:]

        # trim to signal length
        rir = rir[..., :self.signal_length]

        # normalize
        rir = rir / torch.norm(rir, p=2)

        # flip if waveform convolution
        if not self.fft_convolve:
            self.rir = torch.flip(rir, [-1]).reshape(1, 1, -1).to(self.rir)
        else:
            self.rir = rir.reshape(1, 1, -1).to(self.rir)

