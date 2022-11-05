import random
import torch
import torchaudio

import torch.nn.functional as F

from scipy.signal import firwin2

from src.simulation.effect import Effect

torchaudio.set_audio_backend("sox_io")

################################################################################
# Bandpass filter
################################################################################


class Bandpass(Effect):

    def __init__(self, compute_grad: bool = True,
                 low: any = None,
                 high: any = None):
        super().__init__(compute_grad)

        self.min_low, self.max_low = self.parse_range(
            low,
            int,
            f'Invalid cutoff frequency {low}'
        )

        self.min_high, self.max_high = self.parse_range(
            high,
            int,
            f'Invalid cutoff frequency {high}'
        )

        if self.max_high > (self.sample_rate / 2) - 100:
            raise ValueError(
                f'Cutoff too close to Nyquist frequency'
                f' {self.sample_rate/2}Hz; may produce ringing')

        # store impulse response as buffer to allow device movement
        self.low, self.high = None, None
        self.register_buffer("filter", torch.zeros(1, dtype=torch.float32))

        # initialize filter
        self.sample_params()

    def forward(self, x: torch.Tensor):
        """
        Perform waveform convolution with FIR bandpass filter
        """

        # require batch and channel dimensions
        n_batch, signal_length = x.shape[0], x.shape[-1]
        x = x.reshape(n_batch, -1, signal_length)

        pad = F.pad(x, (self.filter.shape[-1]-1, 0))
        return F.conv1d(pad, self.filter.clone().to(x))

    def sample_params(self):
        """
        Sample cutoff frequencies, generate FIR lowpass and highpass filters,
        convolve (with 'full' padding) to obtain a single FIR bandpass filter
        """
        self.low = random.uniform(self.min_low, self.max_low)
        self.high = random.uniform(self.min_high, self.max_high)

        n_taps = 257  # length of each FIR filter
        width = 0.001  # width of filter transition band

        freq_hp = [
            0.0,
            self.low / (1 + width),
            self.low * (1 + width),
            self.sample_rate/2
        ]
        freq_lp = [
            0.0,
            self.high / (1 + width),
            self.high * (1 + width),
            self.sample_rate/2
        ]

        gain_hp = [0.0, 0.0, 1.0, 1.0]
        gain_lp = [1.0, 1.0, 0.0, 0.0]

        hp = torch.as_tensor(
            firwin2(
                numtaps=n_taps,
                freq=freq_hp,
                gain=gain_hp,
                fs=self.sample_rate
            )
        )
        lp = torch.as_tensor(
            firwin2(
                numtaps=n_taps,
                freq=freq_lp,
                gain=gain_lp,
                fs=self.sample_rate
            )
        )

        self.filter = F.conv1d(
            F.pad(
                torch.as_tensor(lp).flip([-1]).reshape(1, 1, -1),
                (hp.shape[-1] - 1, hp.shape[-1] - 1)  # 'full' padding
            ),
            torch.as_tensor(hp).flip([-1]).reshape(1, 1, -1)
        ).flip([-1]).reshape(1, 1, -1).to(self.filter)

