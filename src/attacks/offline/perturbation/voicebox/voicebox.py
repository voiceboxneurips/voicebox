import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import warnings

from pathlib import Path

import numpy as np

from typing import Union, Dict

from torchaudio.transforms import MelScale

from src.constants import PPG_PRETRAINED_PATH
from src.data.dataproperties import DataProperties
from src.attacks.offline.perturbation import Perturbation
from src.models.phoneme import PPGEncoder
from src.attacks.offline.perturbation.voicebox.pitch import PitchEncoder
from src.attacks.offline.perturbation.voicebox.loudness import LoudnessEncoder
from src.attacks.offline.perturbation.voicebox.spec import SpectrogramEncoder
from src.attacks.offline.perturbation.voicebox.bottleneck import (
    RNNBottleneck, CausalTransformer
)
from src.attacks.offline.perturbation.voicebox.lookahead import Lookahead
from src.attacks.offline.perturbation.voicebox.mlp import MLP
from src.attacks.offline.perturbation.voicebox.film import FiLM
from src.attacks.offline.perturbation.voicebox.filter import FilterLayer
from src.attacks.offline.perturbation.voicebox.batchnorm import BatchNorm
from src.attacks.offline.perturbation.voicebox.projection import (
    CausalControlProjection
)

################################################################################
# VoiceBox model for applying adversarial FIR filtering in real-time
################################################################################


class VoiceBox(Perturbation):

    def __init__(self,

                 # encoder topology
                 use_loudness_encoder: bool = True,
                 use_pitch_encoder: bool = True,
                 use_phoneme_encoder: bool = True,
                 use_spec_encoder: bool = True,

                 # SpectrogramEncoder parameters
                 spec_encoder_type: str = 'mel',
                 spec_encoder_n_mels: int = 64,
                 spec_encoder_mlp_depth: int = 2,
                 spec_encoder_hidden_size: int = 512,
                 spec_encoder_lookahead_frames: int = 5,
                 spec_encoder_normalize: str = 'none',

                 # AC-VC encoder parameters
                 ppg_encoder_depth: int = 2,
                 ppg_encoder_hidden_size: int = 512,
                 ppg_encoder_path: Union[str, Path] = PPG_PRETRAINED_PATH,

                 # bottleneck layer parameters
                 bottleneck_type: str = 'lstm',  # lstm
                 bottleneck_skip: bool = True,
                 bottleneck_depth: int = 2,  # 8
                 bottleneck_hidden_size: int = 512,
                 bottleneck_feedforward_size: int = 2048,

                 # optionally, concatenate conditioning information before bottleneck
                 conditioning_dim: int = 0,

                 # post-bottleneck lookahead convolution
                 bottleneck_lookahead_frames: int = 0,

                 # filter control constraint parameters
                 mel_scale_controls: bool = False,
                 neutral_below_hz: float = None,
                 neutral_above_hz: float = None,
                 control_scaling_fn: str = 'sigmoid',
                 control_eps: float = None,
                 projection_norm: Union[str, int, float] = None,
                 projection_context: int = 10,
                 projection_method: str = None,
                 projection_decay: float = 2.0,

                 # FilterLayer parameters
                 n_bands: int = 128,
                 win_length: int = 256,
                 win_type: str = 'hann',
                 normalize_ir: Union[str, int] = None,

                 # audio normalization
                 normalize_audio: str = 'peak',
                 ):

        super().__init__()

        # round window length to next power of 2
        next_pow_2 = 2**(win_length - 1).bit_length()
        if win_length != next_pow_2:
            warnings.warn(f'Rounding block size {win_length} to nearest power'
                          f' of 2 ({next_pow_2})')

        # store attributes
        self.win_length = next_pow_2
        self.win_type = win_type
        self.normalize_audio = normalize_audio
        self.n_bands = n_bands
        self.mel_scale_controls = mel_scale_controls
        self.bottleneck_type = bottleneck_type
        self.bottleneck_skip = bottleneck_skip
        self.bottleneck_depth = bottleneck_depth
        self.bottleneck_hidden_size = bottleneck_hidden_size
        self.bottleneck_feedforward_size = bottleneck_feedforward_size
        self.bottleneck_lookahead_frames = bottleneck_lookahead_frames
        self.spec_encoder_mlp_depth = spec_encoder_mlp_depth
        self.ppg_encoder_hidden_size = ppg_encoder_hidden_size
        self.ppg_encoder_path = ppg_encoder_path

        # ensure at least one encoder network is present
        assert any([
            use_loudness_encoder,
            use_pitch_encoder,
            use_phoneme_encoder,
            use_spec_encoder]), \
            f'Must specify at least one encoder network'

        self.use_loudness_encoder = use_loudness_encoder
        self.use_pitch_encoder = use_pitch_encoder
        self.use_phoneme_encoder = use_phoneme_encoder
        self.use_spec_encoder = use_spec_encoder

        ########################################################################
        # AC-VC ENCODER
        ########################################################################

        if use_phoneme_encoder:
            # AC-VC PPG encoder network
            self.ppg_encoder = PPGEncoder(
                win_length=win_length, 
                hop_length=win_length//2,
                win_func=torch.hann_window,
                n_mels=32,
                n_mfcc=19,
                lstm_depth=ppg_encoder_depth,
                hidden_size=ppg_encoder_hidden_size
            )
            self.ppg_encoder.load_state_dict(
                torch.load(ppg_encoder_path, map_location=torch.device('cpu'))
            )
        else:
            self.ppg_encoder = nn.Identity()

        # AC-VC pitch encoder network
        self.pitch_encoder = PitchEncoder(
            hop_length=win_length//2) if use_pitch_encoder else nn.Identity()

        # A-weighted loudness encoder
        self.loudness_encoder = LoudnessEncoder(
            hop_length=win_length//2) if use_loudness_encoder else nn.Identity()

        # freeze gradients
        for p in self.ppg_encoder.parameters():
            p.requires_grad = False
        for p in self.pitch_encoder.parameters():
            p.requires_grad = False
        for p in self.loudness_encoder.parameters():
            p.requires_grad = False

        # merge AC-VC encoder & spectrogram encoder output
        n_encoder_feats = int(use_phoneme_encoder) * ppg_encoder_hidden_size
        n_encoder_feats += int(use_loudness_encoder) * 1
        n_encoder_feats += int(use_pitch_encoder) * 2
        n_encoder_feats += int(use_spec_encoder) * spec_encoder_hidden_size
        self.encoder_proj = nn.Sequential(
            BatchNorm(num_features=n_encoder_feats),
            nn.Linear(n_encoder_feats, bottleneck_hidden_size),
            nn.ReLU()
        )

        ########################################################################
        # "LOOKAHEAD" SPECTROGRAM ENCODER
        ########################################################################

        # spectrogram encoder, with optional lookahead
        self.spec_encoder = SpectrogramEncoder(
            spec_type=spec_encoder_type,
            n_mels=spec_encoder_n_mels,
            win_length=win_length,
            win_type=win_type,
            lookahead=spec_encoder_lookahead_frames,
            hidden_size=spec_encoder_hidden_size,
            mlp_depth=spec_encoder_mlp_depth,
            normalize=spec_encoder_normalize
        ) if use_spec_encoder else nn.Identity()

        ########################################################################
        # TARGET CONDITIONING
        ########################################################################

        if conditioning_dim > 0:

            self.conditioning_mlp = MLP(
                    in_channels=conditioning_dim,
                    hidden_size=conditioning_dim,
                    depth=2
                )
            self.conditioning_encoder = FiLM(
                        cond_dim=conditioning_dim,
                        num_features=bottleneck_hidden_size,
                        batch_norm=True
                    )

        self.conditioning_dim = conditioning_dim

        ########################################################################
        # LATENT BOTTLENECK
        ########################################################################

        if bottleneck_type in ['lstm', 'rnn']:
            self.bottleneck = RNNBottleneck(
                input_size=bottleneck_hidden_size,
                hidden_size=bottleneck_feedforward_size,
                proj_size=bottleneck_hidden_size,
                num_layers=bottleneck_depth,
                downsample_index=1,
                downsample_factor=1,
                dropout_prob=0.0
            )
        elif bottleneck_type in ['attention', 'transformer']:
            self.bottleneck = CausalTransformer(
                hidden_size=bottleneck_hidden_size,
                dim_feedforward=bottleneck_feedforward_size,
                depth=bottleneck_depth,
                heads=8,
                dropout_prob=0.0
            )
        else:
            self.bottleneck = nn.Identity()

        # post-bottleneck projection with optional lookahead
        n_bottleneck_feats = 2 * bottleneck_hidden_size if bottleneck_skip \
            else bottleneck_hidden_size
        self.bottleneck_proj = nn.Sequential(
            Lookahead(
                n_features=n_bottleneck_feats,
                lookahead_frames=bottleneck_lookahead_frames
            ) if bottleneck_lookahead_frames else nn.Identity(),
            nn.LeakyReLU() if bottleneck_lookahead_frames else nn.Identity(),
            BatchNorm(n_bottleneck_feats),
            nn.Linear(n_bottleneck_feats, n_bands)
        )

        ########################################################################
        # DECODER
        ########################################################################

        assert control_scaling_fn.lower() in ['sigmoid', 'elu', 'relu', 'log'], \
            f'Invalid filter control scaling function {control_scaling_fn}'
        self.control_scaling_fn = control_scaling_fn

        # (optionally-causal) filter-control projection module
        self.projector = CausalControlProjection(
            eps=control_eps,
            n_controls=n_bands,
            unity=0.0 if control_scaling_fn == 'log' else 1.0,
            projection_norm=projection_norm,
            method=projection_method,
            decay=projection_decay,
            context=projection_context
        )

        # constrain filter controls
        self.cutoff_high = self.hz_to_band(
            neutral_above_hz,
            n_bands,
            DataProperties.get('sample_rate')
        ) if neutral_above_hz else n_bands
        self.cutoff_low = self.hz_to_band(
            neutral_below_hz,
            n_bands,
            DataProperties.get('sample_rate')
        ) if neutral_below_hz else 0

        # optionally, use mel-scaled filter controls
        self.register_buffer("inv_mel_fb", MelScale(
            n_mels=n_bands,
            sample_rate=DataProperties.get('sample_rate'),
            n_stft=n_bands
        ).fb.transpose(0, 1).pinverse())
        self.inv_mel_scale = lambda x: torch.matmul(
            self.inv_mel_fb,
            x.permute(0, 2, 1)
        ).clamp(min=0, max=None).permute(0, 2, 1)

        # filter
        self.filter = FilterLayer(
            win_length=win_length,
            win_type=win_type,
            n_bands=n_bands,
            normalize_ir=normalize_ir
        )

        # references for visualization
        self.ref_wav = torch.empty(0)
        self.ref_controls = torch.empty(0)

    def set_reference(self, x: torch.Tensor):
        """Store reference audio for visualization/logging"""

        # ensure the input contains at least a single frame of audio
        assert x.shape[-1] >= self.win_length

        # avoid modifying input audio
        x = x.clone().detach()
        n_batch, *channel_dims, signal_len = x.shape

        # add channel dimension if necessary
        if len(channel_dims) == 0:
            x = x.unsqueeze(1)

        # store reference audio
        self.ref_wav = x[0]

        # store reference controls
        with torch.no_grad():
            self.ref_controls = self.get_controls(x)[0]

    def _project_valid_top_level(self):
        pass

    def _visualize_top_level(self) -> Dict[str, torch.Tensor]:
        """Visualize controls and reference audio"""

        name = self.__class__.__name__

        visualizations = {}

        # compute controls for stored reference audio
        with torch.no_grad():
            self.ref_controls = self.get_controls(
                self.ref_wav
            )

        def band_to_hz(band: int):
            nyquist = DataProperties.get('sample_rate') // 2
            hz = int(band / self.n_bands * nyquist)
            return hz

        # plot controls
        import io
        import matplotlib.pyplot as plt
        from PIL import Image
        from torchvision.transforms import ToTensor

        _, t, f = self.ref_controls.shape

        # scale controls
        if self.control_scaling_fn.lower() == 'sigmoid':
            controls = 2 * torch.sigmoid(
                self.ref_controls
            )**(math.log(10)) + 1e-7
        elif self.control_scaling_fn.lower() == 'relu':
            controls = F.relu(self.ref_controls)
        elif self.control_scaling_fn.lower() == 'elu':
            controls = F.elu(self.ref_controls) + 1
        elif self.control_scaling_fn.lower() == 'log':
            controls = torch.tanh(self.ref_controls * 0.2) * 5  # [-5, 5]
            controls = torch.exp(controls)
        else:
            raise ValueError(f'Invalid control scaling function '
                             f'{self.control_scaling_fn}')

        # if controls are taken to be mel-scaled, linearly scale
        if self.mel_scale_controls:
            controls = self.inv_mel_scale(controls)

        # perform projection (PGD step) on filter controls
        controls = self._project_valid(controls)

        # scale to [-4, 4] for plotting, with 0 at center (logarithmic)
        # clips near zero, and
        controls = torch.clamp(torch.log2(controls + 1e-8), min=-4, max=4)
        controls = controls.clone().detach().squeeze().cpu().numpy()

        fig, axs = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(30, 30),
            gridspec_kw={'height_ratios': [4, 1]})

        # draw frame boundaries
        for frame in range(t):
            axs[0].vlines(
                frame,
                ymin=-10,
                ymax=f * 10,
                color='k',
                alpha=0.0,
                linewidth=.5)

        for band in range(f):
            axs[0].plot([band * 10]*t, 'k', alpha=0.6, linewidth=.5)
            axs[0].plot(controls[:, band] + band * 10, alpha=0.9, linewidth=2)
            axs[0].set_ylabel('Filter Band', fontsize=30)

        axs[0].set_yticks(
            [10 * i for i in range(f)],
            [f'{band_to_hz(i)}Hz' for i in range(f)])

        axs[1].plot(self.ref_wav.cpu().numpy().flatten()[::25], color='k', linewidth=1)
        axs[1].set_ylabel('Waveform Amplitude', fontsize=30)
        axs[1].set_ylim([-1, 1])
        plt.tight_layout()

        # save plot to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)

        # return plot image as tensor
        output = ToTensor()(np.array(img))

        visualizations = {
            **visualizations,
            f'{name}-parameters': output
        }

        return visualizations

    @staticmethod
    def hz_to_band(f: float, n_bands: int, sr: int):
        nyquist = sr // 2
        band = max(0,
                   min(
                       int(f * n_bands / nyquist), n_bands
                   ))
        return band

    def _project_valid(self, controls: torch.Tensor):

        # account for log-scaling
        if self.control_scaling_fn.lower() == 'log':
            controls = torch.log(controls + 1e-8)

        controls = self.projector(controls)

        # account for log-scaling
        if self.control_scaling_fn == 'log':
            controls = torch.exp(controls)

        # if specified, keep filter neutral in given ranges
        if self.cutoff_low:
            # avoid in-place operations in forward pass
            b, t, c = controls.shape
            controls = torch.cat(
                [
                    torch.ones((b, t, self.cutoff_low), device=controls.device),
                    controls[..., self.cutoff_low:],
                ], dim=-1)
            assert controls.shape == (b, t, c)

        if self.cutoff_high:
            # avoid in-place operations in forward pass
            b, t, c = controls.shape
            controls = torch.cat(
                [
                    controls[..., :self.cutoff_high],
                    torch.ones((b, t, c - self.cutoff_high), device=controls.device),
                ], dim=-1)
            assert controls.shape == (b, t, c)

        return controls

    def get_controls(self,
                     x: torch.Tensor,
                     pitch: torch.Tensor = None,
                     periodicity: torch.Tensor = None,
                     loudness: torch.Tensor = None,
                     y: torch.Tensor = None,
                     *args, **kwargs
                     ):
        """Map audio inputs to frame-wise filter controls"""

        # ensure the input contains at least a single frame of audio
        assert x.shape[-1] >= self.win_length

        ########################################################################
        # AC-VC ENCODER
        ########################################################################

        with torch.no_grad():

            features = []

            # compute features if necessary
            if self.use_pitch_encoder:
                if pitch is None or periodicity is None:
                    pitch, periodicity = self.pitch_encoder(x)
                features.extend([pitch, periodicity])

            if self.use_loudness_encoder:
                if loudness is None:
                    loudness = self.loudness_encoder(x)
                features.append(loudness)

            # compute phonetic posteriorgrams (PPGs)
            if self.use_phoneme_encoder:
                ppg = self.ppg_encoder(x)
                features.append(ppg)

        ########################################################################
        # SPECTROGRAM ENCODER
        ########################################################################

        if self.use_spec_encoder:
            spec = self.spec_encoder(x)  # (n_batch, n_frames, hidden_size)
            features.append(spec)

        ########################################################################
        # MERGE ENCODINGS
        ########################################################################

        encoded = self.encoder_proj(
            torch.cat(features, dim=-1))  # (n_batch, n_frames, hidden_size)

        ########################################################################
        # TARGET CONDITIONING
        ########################################################################

        if self.conditioning_dim:

            n_frames = encoded.shape[1]

            if y is not None:
                assert y.shape[-1] == self.conditioning_dim
                assert y.ndim == 3

                # average over all segments if present
                y = y.mean(dim=1, keepdim=True)

                # duplicate over all frames
                y = y.repeat(1, n_frames, 1)

            else:

                y = torch.zeros(
                    (x.shape[0], n_frames, self.conditioning_dim),
                    device=x.device)

            # apply learned affine transformations to feature dimension
            encoded = self.conditioning_encoder(
                x=encoded,
                cond=self.conditioning_mlp(y)
            )  # (n_batch, n_frames, hidden_size)

        ########################################################################
        # BOTTLENECK
        ########################################################################

        bottleneck_out = self.bottleneck(encoded)

        # apply skip connection with pre-bottleneck encoding
        if self.bottleneck_skip:
            bottleneck_out = torch.cat([
                bottleneck_out,
                encoded
            ], dim=-1)  # (n_batch, n_frames, 2 * hidden_size)

        controls = self.bottleneck_proj(bottleneck_out)  # (n_batch, n_frames, n_bands)

        return controls

    def forward(self,
                x: torch.Tensor,
                pitch: torch.Tensor = None,
                periodicity: torch.Tensor = None,
                loudness: torch.Tensor = None,
                y: torch.Tensor = None,
                *args, **kwargs):

        # ensure the input contains at least a single frame of audio
        assert x.shape[-1] >= self.win_length

        # require batch, channel dimensions
        assert x.ndim >= 2
        n_batch, *channel_dims, signal_len = x.shape

        if x.ndim == 2:
            x = x.unsqueeze(1)

        # convert to mono audio
        x = x.mean(dim=1, keepdim=True)

        # if features are provided, check dimensions
        assert pitch is None or pitch.shape[0] == x.shape[0]
        assert periodicity is None or periodicity.shape[0] == x.shape[0]
        assert loudness is None or loudness.shape[0] == x.shape[0]

        # prepare to normalize output volume
        peak = torch.max(torch.abs(x), -1)[0].reshape(n_batch)

        controls = self.get_controls(x,
                                     pitch,
                                     periodicity,
                                     loudness,
                                     y)  # (n_batch, n_frames, n_bands)

        ########################################################################
        # CONTROL SCALING
        ########################################################################

        # scale stored controls to fixed range
        if self.control_scaling_fn.lower() == 'sigmoid':
            controls = 2 * torch.sigmoid(
                controls
            )**(math.log(10)) + 1e-7
        elif self.control_scaling_fn.lower() == 'relu':
            controls = F.relu(controls)
        elif self.control_scaling_fn.lower() == 'elu':
            controls = F.elu(controls) + 1
        elif self.control_scaling_fn.lower() == 'log':
            controls = torch.tanh(controls * 0.2) * 5  # [-5, 5]
            controls = torch.exp(controls)
        else:
            raise ValueError(f'Invalid control scaling function '
                             f'{self.control_scaling_fn}')

        # if controls are taken to be mel-scaled, linearly scale
        if self.mel_scale_controls:
            controls = self.inv_mel_scale(controls)

        # perform projection (PGD step) on filter controls
        controls = self._project_valid(controls)

        ########################################################################
        # FILTERING
        ########################################################################

        # apply filter
        x = self.filter(x, controls)

        # apply normalization to match input volume
        if self.normalize_audio in [None, 'none']:
            factor = 1.0
        elif self.normalize_audio == 'peak':
            factor = peak / (torch.max(torch.abs(x), -1)[0].reshape(n_batch) + 1e-6)
            factor = factor.reshape(n_batch, 1, 1)
        else:
            raise ValueError(f'Invalid audio normalization type {self.normalize_audio}')

        x = x * factor

        # match original dimensions
        x = x[..., :signal_len].reshape(n_batch, *((1,) * len(channel_dims)), signal_len)

        return x
