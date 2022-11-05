import random
import math
import torch
import torch.nn.functional as F
import torchaudio

from pathlib import Path

import librosa as li
from src.simulation.effect import Effect

torchaudio.set_audio_backend("sox_io")

################################################################################
# Simulate environmental noise
################################################################################


class Noise(Effect):
    """
    Simple additive noise effect
    """
    def __init__(self,
                 compute_grad: bool = True,
                 type: str = 'gaussian',
                 snr: any = None,
                 noise_dir: str = None,
                 ext: str = "wav"):
        """
        Apply additive noise to audio signal. SNR calculations adapted from
        VoxCeleb-Trainer (https://github.com/clovaai/voxceleb_trainer/)

        :param compute_grad: if False, perform straight-through gradient
                             estimation
        :param type: type of noise to add; must be one of `gaussian`,
                     `uniform`, or `environmental`
        :param snr: decibel Signal-to-Noise ratio (dB SNR) of added noise
        :param noise_dir: directory from which to draw noise samples, if `type`
                          is `environmental`
        :param ext: extension for audio files in `noise_dir`
        """
        super().__init__(compute_grad)

        self.type = type
        self.noise_list = None
        self.ext = ext

        if type == 'environmental':
            if not noise_dir:
                raise ValueError(
                    'Environmental noise requires sample directory'
                )
            else:
                self.noise_list = list(Path(noise_dir).rglob(f'*.{self.ext}'))

        # parse valid range of SNR parameter
        self.min_snr, self.max_snr = self.parse_range(
            snr,
            float,
            f'Invalid noise SNR {snr}'
        )

        # store noise as buffer to allow device movement
        self.register_buffer("noise", torch.zeros(1, dtype=torch.float32))
        self.register_buffer("noise_db", torch.zeros(1, dtype=torch.float32))

        # initialize parameters
        self.snr = None
        self.sample_params()

    def forward(self, x: torch.Tensor):

        # require batch, channel dimensions
        assert x.ndim >= 2
        orig_shape = x.shape

        if x.ndim == 2:
            x = x.unsqueeze(1)

        # scale noise level to stored SNR
        signal_db = 10 * torch.log10(
            torch.mean(torch.square(x), dim=-1, keepdims=True) + 1e-8
        )
        scale = torch.sqrt(
            torch.pow(10, (signal_db - self.noise_db - self.snr) / 10)
        )

        # scale noise and trim to input length
        noise = scale * self.noise.clone().to(x)[..., :x.shape[-1]]

        # repeat noise to match input length if necessary
        pad_len = max(x.shape[-1] - noise.shape[-1], 0)
        noise = F.pad(noise, (0, pad_len), mode='circular')

        # reshape to original dimensions
        return (noise + x).reshape(orig_shape)

    @staticmethod
    def _crossfade(sig, fade_len):
        sig = sig.clone()
        fade_len = int(fade_len * sig.shape[-1])
        fade_in = torch.linspace(0, 1, fade_len).to(sig)
        fade_out = torch.linspace(1, 0, fade_len).to(sig)
        sig[..., :fade_len] *= fade_in
        sig[..., -fade_len:] *= fade_out
        return sig

    def sample_params(self):
        """
        Sample SNR uniformly from stored range
        """
        self.snr = random.uniform(self.min_snr, self.max_snr)

        if self.type == "gaussian":
            self.noise = torch.randn(self.signal_length).to(self.noise)
        elif self.type == "uniform":
            self.noise = torch.sign(
                torch.randn(self.signal_length)
            ).to(self.noise)
        elif self.type == "environmental":

            # load from randomly-selected file
            noise_np, _ = li.load(
                random.choice(self.noise_list),
                sr=self.sample_rate, mono=True
            )
            noise = torch.as_tensor(noise_np)

            # trim or loop (with cross-fade) to match expected signal length
            if noise.shape[-1] >= self.signal_length:
                self.noise = noise[..., :self.signal_length].reshape(
                    1, 1, -1
                ).to(self.noise)
            else:

                overlap = 0.05
                step = math.ceil(noise.shape[-1] * (1 - overlap))
                n_repeat = math.ceil(self.signal_length / step)

                padded = torch.zeros(
                    1, step * (n_repeat - 1) + noise.shape[-1] + 1
                ).reshape(1, -1).type(torch.float32)
                shape = padded.shape[:-1] + (n_repeat, noise.shape[-1])

                strides = (padded.stride()[0],) + (step, padded.stride()[-1],)
                frames = torch.as_strided(
                    padded, size=shape, stride=strides
                )[::step]

                for j in range(n_repeat):
                    frames[:, j, :] += self.crossfade(noise, overlap)

                self.noise = padded[..., :self.signal_length].reshape(
                    1, 1, -1
                ).to(self.noise)

        else:
            raise ValueError(f'Invalid noise type {self.type}')

        self.noise_db = 10 * torch.log10(
            torch.mean(torch.square(self.noise), dim=-1, keepdims=True) + 1e-8
        ).to(self.noise_db)
