from typing import Any, Callable, Union, Optional, Tuple
import math
import torchaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MFCC, MelSpectrogram
import pyworld


from src.attacks.offline.perturbation import VoiceBox
from src.attacks.offline.perturbation.voicebox.bottleneck import RNNBottleneck
from src.attacks.offline.perturbation.voicebox.filter import FilterLayer
from src.attacks.offline.perturbation.voicebox.spec import SpectrogramEncoder
from src.models.phoneme import Delta
from src.data import DataProperties

"""
VoiceBoxStreamer: An implementation of VoiceBoxStreamer that works with `src.attacks.online.streamer.Streamer`

Weights can be loaded from a `VoiceBox` object.
"""

class VoiceBoxStreamer(VoiceBox):
    def __init__(self, encoder_buffer_frames=50, bottleneck_type='lstm', **kwargs):
        super().__init__(**kwargs)
        self.hop_length = self.win_length // 2
        self._condition_vector: torch.Tensor = torch.zeros(
            1, 1, self.conditioning_dim
        )
        # overwrite filter layer with streamer
        self.filter = FilterLayerStreamer(
            win_length=self.filter.win_length,
            win_type=self.filter.win_type,
            n_bands=self.filter.n_bands,
            normalize_ir=self.filter.normalize_ir
        )
        self._init_bottleneck(bottleneck_type)
        self._init_encoder_streamer(encoder_buffer_frames)

    @property
    def window_length(self):
        return self.win_length

    @property
    def condition_vector(self):
        return self._condition_vector
    
    @condition_vector.setter
    def condition_vector(self, v: torch.Tensor):
        assert v.shape == self._condition_vector.shape
        self._condition_vector = v

    def _init_bottleneck(self, bottleneck_type) -> None:
        """
        Initializes Bottleneck. If 'rnn' or 'lstm', initialized with
        a streaming version is used. Transformer is not supported with the streamer.
        """
        if bottleneck_type in ['lstm', 'rnn']:
            self.bottleneck = RNNBottleneckStreamer(
                input_size=self.bottleneck_hidden_size,
                hidden_size=self.bottleneck_feedforward_size,
                proj_size=self.bottleneck_hidden_size,
                num_layers=self.bottleneck_depth,
                downsample_index=1,
                downsample_factor=1,
                dropout_prob=0.0
            )
        elif bottleneck_type in ['attention', 'transformer']:
            raise NotImplementedError("Not Supported.")
        else:
            self.bottleneck = nn.Identity()

    def _init_encoder_streamer(self, encoder_buffer_frames: int) -> None:
        """
        Initializes encoder streamer implementations
        """
        if not isinstance(self.ppg_encoder, nn.Identity):
            # NOTE: Assuming that he PPG is relatively constant.
            # Don't touch the magic numbers
            self.ppg_encoder = PPGEncoderStreamer(
                win_length=self.win_length,
                hop_length=self.win_length // 2,
                win_func=torch.hann_window,
                n_mels=32,
                n_mfcc=19,
                lstm_depth=2,
                hidden_size=self.ppg_encoder_hidden_size,
                lookahead=self.bottleneck_lookahead_frames
            )

            self.ppg_encoder.load_state_dict(
                torch.load(self.ppg_encoder_path, map_location=torch.device('cpu'))
            )
        if not isinstance(self.pitch_encoder, nn.Identity):
            pitch_streamer = DioStreamer(
                return_periodicity=True,
                buffer_frames=encoder_buffer_frames,
                lookahead_frames=self.bottleneck_lookahead_frames
                )
            pitch_streamer.__dict__.update(self.pitch_encoder.__dict__)
            self.pitch_encoder = pitch_streamer
        if not isinstance(self.loudness_encoder, nn.Identity):
            self.loudness_encoder = LoudnessEncoderStreamer(
                hop_length=self.win_length // 2
            )
        if not isinstance(self.spec_encoder, nn.Identity):
            spec_streamer_encoder = SpectrogramEncoderStreamer(
                win_length=self.spec_encoder.win_length,
                win_type=self.spec_encoder.win_type,
                spec_type=self.spec_encoder.spec_type,
                n_mels=self.spec_encoder.n_mels,
                lookahead=self.spec_encoder.lookahead,
                hidden_size=self.spec_encoder.hidden_size,
                mlp_depth=self.spec_encoder.mlp_depth,
                normalize=self.spec_encoder.normalize
            )
            self.spec_encoder = spec_streamer_encoder
    
    def get_controls(
        self,
        x: torch.Tensor,
        hx: Any = None) -> tuple[torch.Tensor, Any]:
        """
        Gets controls and recurrent states

        :return controls: Controls tensor of (1, input_windows, n_fft)
        :return hx: Recurrent state. Do not edit, except for feeding back to this function.
        """
        ppg_hx, bottleneck_hx = (None, None) if hx is None else hx
        n_frames = x.shape[1]

        features = []
        # I think theres some shared stft calls between all of these. 

        if self.use_pitch_encoder:
            pitch, periodicity = self.pitch_encoder(x)
            # clip the first `self._buffer_frames` values
            # We clip an extra frame off the DIO results
            #  because the F0 is assigned inclusively to the first and
            #  last sample.
            features += [pitch, periodicity]
        if self.use_loudness_encoder:
            loudness = self.loudness_encoder(x)
            # clip the first `self._buffer_frames` values
            features.append(loudness)
        if self.use_phoneme_encoder:
            ppg, ppg_hx = self.ppg_encoder(x, hx=ppg_hx)
            features.append(ppg)
        if self.use_spec_encoder:
            spec = self.spec_encoder(x)  # (n_batch, n_frames, hidden_size)
            features.append(spec)

        encoded = self.encoder_proj(
            torch.cat(features, dim=-1)
        )
        ########################################################################
        # TARGET CONDITIONING
        ########################################################################
        if self.conditioning_dim:
            y = self.condition_vector.repeat(1, n_frames, 1)
            encoded = self.conditioning_encoder(x=encoded, cond=self.conditioning_mlp(y))
        # (1, n_frames, hidden_size)

        ########################################################################
        # BOTTLENECK
        ########################################################################
        bottleneck_out, bottleneck_hx = self.bottleneck(
            encoded, hx=bottleneck_hx, lookahead=self.bottleneck_lookahead_frames)

        # apply skip connection with pre-bottleneck encoding
        if self.bottleneck_skip:
            bottleneck_out = torch.cat([
                bottleneck_out,
                encoded
            ], dim=-1)  # (n_batch, n_frames, 2 * hidden_size)

        controls = self.bottleneck_proj(bottleneck_out)  # (n_batch, n_frames, n_bands)
        return controls, (ppg_hx, bottleneck_hx)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, hx: Any=None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Expecting: x: (1, input_windows, window_length)

        This should never get a batch size greater than 1

        Hidden state should not be edited outside of this function
        """
        assert not self.training, "Never use this streamer for training!"
        assert x.shape[-1] == self.win_length
        assert x.shape[0] == 1, "Batched audio not supported"
        assert x.shape[1] > 1, "Due to what I believe to be a Pytorch bug, " \
            + "InstanceNorm1d does not allow single batch single frame inputs, " \
            + "even when `model.eval()` is called beforehand. Please use at least 2 frames at a time."

        controls, hx = self.get_controls(x, hx=hx)
        x, controls = self.chop_lookahead(x, controls)
        controls = self.controls_scaling(controls)
        filtered_audio = self.filter(x, controls)
        return filtered_audio, hx

    def chop_lookahead(self, x: torch.Tensor, controls: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lookahead = self.spec_encoder.lookahead + self.bottleneck_lookahead_frames
        return x[:, :-lookahead, :], controls[:, :-lookahead, :]

    def controls_scaling(self, controls: torch.Tensor) -> torch.Tensor:

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
        return controls

###############################################
#         STREAMING BUILDING BLOCKS           #
###############################################


class RNNBottleneckStreamer(RNNBottleneck):
    """
    Edited Bottleneck to return the hidden states.
    """
    def forward(self,
                x: torch.Tensor,
                hx: Any=None,
                lookahead: int=0
                ) -> tuple[torch.Tensor, Any]:
        if hx is None:
            hx = [None] * self.num_layers
        else:
            assert len(hx) == self.num_layers
        for i, rnn_layer in enumerate(self.rnn):
            x, hx[i] = self.run_rnn(rnn_layer, (x, hx[i]), lookahead)
            x = self.dropout(x)

            # x = self.instancenorm(x.permute(0, 2, 1)).permute(0, 2, 1)
            if i == self.downsample_index:
                n_batch, n_frames, proj_size = x.shape
                # determine necessary padding to allow temporal downsampling
                pad_len = self.downsample_factor * math.ceil(n_frames / self.downsample_factor) - n_frames
                # apply causal padding
                x = F.pad(x, (0, 0, 0, pad_len))
                # apply temporal downsampling
                x = torch.reshape(x, (n_batch, x.shape[1] // self.downsample_factor, x.shape[2] * self.downsample_factor))
        return x, hx

    @staticmethod
    def run_rnn(rnn_layer: nn.Module, data: tuple[torch.Tensor, Any], lookahead: int) -> tuple[torch.Tensor, Any]:
        x, hx = data
        if lookahead == 0:
            return rnn_layer(x, hx)
        else:
            num_frames = x.shape[1]
            if num_frames <= lookahead:
                x, _  = rnn_layer(x, hx)
                return x, hx
            else:
                x, x_l = x[:, :-lookahead, :], x[:, -lookahead:, :]
                x, hx = rnn_layer(x, hx)
                x_l, _ = rnn_layer(x_l, hx)
                x = torch.cat([x, x_l], dim=1)
                return x, hx


class FilterLayerStreamer(FilterLayer):
    """
    Streamer Implementation of FilterLayer without the OLA
    """
    def __init__(self, win_length: int = 512, win_type: str = 'hann', n_bands: int = 128, normalize_ir: Union[str, int, float] = None, **kwargs):
        super().__init__(win_length, win_type, n_bands, normalize_ir, **kwargs)

        # Why is computed every iteration
        n_fft_min = self.win_length + 2 * (self.n_bands - 1)
        self.n_fft = pow(2, math.ceil(math.log2(n_fft_min))) 

    def forward(self, x: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        """

        :param x: Windowed audio of size (1, input_windows, window_size)
        :param controls: Controls of size (1, input_windows, n_fft)
        """
        if x.shape[1] == 0:
            return x

        impulse = self._amp_to_ir(controls)
        x = self._fft_convolve(x, impulse, self.n_fft).contiguous()
        # x: (1, n_frames, n_fft)
        return x

###############################################
#         Streaming Encoding Blocks           #
###############################################

class PPGEncoderStreamer(nn.Module):
    def __init__(self,
                 win_length: int = 256,
                 hop_length: int = 128,
                 win_func: Callable = torch.hann_window,
                 n_mels: int = 32,
                 n_mfcc: int = 13,
                 lstm_depth: int = 2,
                 hidden_size: int = 512,
                 lookahead: int = 5):
        
        super().__init__()
        self.win_length = win_length
        self.hop_length = hop_length
        # get non-center MFCC
        mel_kwargs = {
            "n_fft": self.win_length,
            "win_length": self.win_length,
            "hop_length": self.hop_length,
            "window_fn": win_func,
            "n_mels": n_mels
        }
        spectrogram = NonCenterSpectrogram(
            n_fft=self.win_length,
            win_length=self.win_length,
            hop_length=self.hop_length,
            window_fn=win_func
        )
        self.mfcc = MFCC(
            sample_rate=DataProperties.get("sample_rate"),
            n_mfcc=n_mfcc,
            log_mels=True,
            melkwargs=mel_kwargs
        )
        self.mfcc.MelSpectrogram.spectrogram = spectrogram
        # compute first- and second- order MFCC deltas
        self.delta = Delta()

        # PPG network
        self.mlp = nn.Sequential(
            nn.Linear(n_mfcc * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_depth,
            bias=True,
            batch_first=True,
            bidirectional=False
        )
        self.lookahead = lookahead


    @torch.no_grad()
    def forward(self, x: torch.Tensor, hx: Optional[Tuple[torch.Tensor, torch.Tensor]]=None) -> Tuple[torch.Tensor, Any]:
        """
        :param x: Windowed audio of (1, windows, win_length)
        """
        # require batch, channel dimensions
        mfcc = self.mfcc(x)  # (1, n_mfcc, n_frames)

        delta1 = self.delta(mfcc)  # (1, n_mfcc, n_frames)
        delta2 = self.delta(delta1)  # (1, n_mfcc, n_frames)
        x = torch.cat([mfcc, delta1, delta2], dim=1)  # (n_batch, n_frames, 3 * n_mfcc)
        x = x.permute(0, 2, 1)
        x = self.mlp(x)  # (n_batch, n_frames, hidden_size)
        if self.lookahead:
            if x.shape[1] > self.lookahead:
                x, x_l = x[:, :-self.lookahead, :], x[:, -self.lookahead:, :]
                x, hx = self.lstm(x, hx)  # (n_batch, n_frames, hidden_size)
                x_l, _ = self.lstm(x_l, hx)
                x = torch.cat([x, x_l], dim=1)
            else:
                x, _ = self.lstm(x, hx)
        else:
            x, hx = self.lstm(x, hx)

        return x, hx

class DioStreamer(nn.Module):
    """
    Pitch and Periodicity streamer.

    Only uses dio.
    """
    def __init__(
        self,
        return_periodicity: bool=True,
        hop_length: int=128,
        buffer_frames: int = 20,
        lookahead_frames: int = 5):
        self.return_periodicity = return_periodicity
        self.hop_length = hop_length

        self._buffer_frames = buffer_frames
        self._lookahead = lookahead_frames
        self._buffer = torch.zeros(1, buffer_frames, self.hop_length * 2)

    def _roll_buffer(self, last_input_frames: torch.Tensor) -> None:
        if last_input_frames.shape[1] > self._lookahead:
            self._buffer = (
                torch.cat(
                    [self._buffer, last_input_frames],
                    dim=1
                )[:, -self._buffer_frames-self._lookahead:-self._lookahead, :]
            )

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """
        Takes windowed audio. Reconstructs the overlapping frames
        by taking the first `hop_length` frames of each window, and stitching.
        Them back together. Dio returns estimated pitch and periodicity 
        in hop_size distances

        """
        num_windows = x.shape[1] + self._buffer_frames
        pitch_out, peridoicity_out, device = [], [], x.device
        hop_ms = 1000 * self.hop_length / DataProperties.get('sample_rate')
        x_padded = torch.cat([self._buffer, x], dim=1)
        x_folded = F.fold(
            x_padded[..., :self.hop_length].permute(0, 2, 1),
            output_size=(1, self.hop_length * num_windows),
            kernel_size=(1, self.hop_length),
            stride=(1, self.hop_length)
        )
        x_folded = x_folded.flatten()
        x_folded = torch.cat([x_folded, x[0, -1, self.hop_length:]])
        # x_folded: (self.hop_length * num_windows)
        x_np = x_folded.clone().double().cpu().numpy()


        pitch, timeaxis = pyworld.dio(
            x_np,
            fs=DataProperties.get('sample_rate'),
            f0_floor=50,
            f0_ceil=550,
            frame_period=hop_ms,
            allowed_range=.1,
            speed=4)  # downsampling factor, for speedup
        pitch = pyworld.stonemask(
            x_np,
            pitch,
            timeaxis,
            DataProperties.get('sample_rate'))

        pitch_out.append(pitch)

        pitch_out = torch.as_tensor(
            pitch_out,
            dtype=torch.float32,
            device=device).unsqueeze(-1)

        pitch_out = pitch_out[:, self._buffer_frames+1:-1, :]
        out = pitch_out
        if self.return_periodicity:
            unvoiced = pyworld.d4c(
                x_np,
                pitch,
                timeaxis,
                DataProperties.get('sample_rate'),
            ).mean(axis=1)

            peridoicity_out.append(unvoiced)
            periodicity_out = torch.as_tensor(
                peridoicity_out,
                dtype=torch.float32,
                device=device).unsqueeze(-1)

            # (n_batch, n_frames, 1), (n_batch, n_frames, 1)
            periodicity_out = periodicity_out[:, self._buffer_frames+1:-1, :]
            out = pitch_out, periodicity_out
        self._roll_buffer(x)
        return out


class LoudnessEncoderStreamer(nn.Module):
    """Streaming implementation for LoudnessEncoder"""
    def __init__(self,
                 hop_length: int = 128,
                 n_fft: int = 256) -> None:
        super().__init__()
        self.hop_length = hop_length
        self.n_fft = n_fft

    def A_weight(self):
        # torch implementation of A_weight
        # librosa.fft_frequencies
        freqs = torch.zeros(1 + self.n_fft // 2)
        for i in range(1 + self.n_fft // 2):
            freqs[i] = i * DataProperties.get('sample_rate') / self.n_fft

        # librosa.A_weighting. Using magic numbers from librosa implementation
        min_db = torch.Tensor([-80.0]).float()
        f_sq = freqs ** 2.0
        const = torch.Tensor([12194.217, 20.598997, 107.65265, 737.86223]) ** 2.0
        weights = 2.0 + 20.0 * (
            torch.log10(const[0])
            + 2 * torch.log10(f_sq)
            - torch.log10(f_sq + const[0])
            - torch.log10(f_sq + const[1])
            - 0.5 * torch.log10(f_sq + const[2])
            - 0.5 * torch.log10(f_sq + const[3])
        )
        return torch.where(weights > min_db, weights, min_db)

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """
        :param x:  Windowed audio of (1, num_windows, window_length)
        """
        # torch.stft should be exactly the same as librosa stft
        spec = torch.stft(
            x.squeeze(0),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=torch.hann_window(self.n_fft),
            center=False,
            return_complex=True
        ).permute(2, 1, 0) # x: (1, num_windows, n_fft)

        spec = torch.log(abs(spec) + 1e-7)
        a_weight = self.A_weight()
        # apply multiplicative weighting via addition in log domain
        spec = spec + a_weight.reshape(1, -1, 1)

        # take mean over each frame
        loudness = torch.mean(spec, dim=1).unsqueeze(-1).float().to(x.device)

        return loudness

class SpectrogramEncoderStreamer(SpectrogramEncoder):
    """
    Streaming Implementation of SpectrogramEncoder
    
    Takes windowed audio instead of a single signal.
    """
    def __init__(
        self, 
        win_length: int = 512, 
        win_type: str = 'hann', 
        spec_type: str = 'linear', 
        lookahead: int = 5,
        hidden_size: int = 512,
        n_mels: int = 64,
        mlp_depth: int = 2,
        normalize: str = None
    ):
        super().__init__(win_length, win_type, spec_type, lookahead, hidden_size, n_mels, mlp_depth, normalize)

        # compute spectral representation
        spec_kwargs = {
            "n_fft": self.win_length,
            "win_length": self.win_length,
            "hop_length": self.hop_length,
            "window_fn": self._get_win_func(self.win_type)
        }
        spectrogram = NonCenterSpectrogram(**spec_kwargs)
        mel_kwargs = {**spec_kwargs, "n_mels": self.n_mels}

        if spec_type == 'linear':
            self.spec = spectrogram
        elif spec_type == 'mel':
            self.spec = MelSpectrogram(
                sample_rate=DataProperties.get("sample_rate"),
                **mel_kwargs
            )
            self.spec.spectrogram = spectrogram
        elif spec_type == 'mfcc':
            self.spec = MFCC(
                sample_rate=DataProperties.get("sample_rate"),
                n_mfcc=self.n_mels,
                log_mels=True,
                melkwargs=mel_kwargs
            )
            self.spec.MelSpectrogram.spectrogram = spectrogram

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # spec = self.spec(x).squeeze(-1) + 1e-6  # (n_batch, n_frames, n_freq)
        spec = self.spec(x) + 1e-6  # (n_batch, n_freqs, n_frames)

        if self.spec_type in ['linear', 'mel']:
            spec = 10 * torch.log10(spec + 1e-8)  # (n_batch, n_freq, n_frames)
        # normalize spectrogram
        spec = self.norm(spec)  # (n_batch, n_freq, n_frames)

        # actual encoder network
        encoded = self.glu(spec)  # (n_batch, hidden_size, n_frames)
        encoded = self.conv(encoded)  # (n_batch, hidden_size, n_frames)
        encoded = self.mlp(
            encoded.permute(0, 2, 1)
        )  # (n_batch, n_frames, hidden_size)

        return encoded

class NonCenterSpectrogram(torchaudio.transforms.Spectrogram):
    """
    A modified Spectrogram Module that processes overlapping frames
    of audio in shape (batch, num_frames, frame_length) to
    (batch, n_fft, num_frames).

    This should be used to patch the MelSpectrogram's `spectrogram`.

    Why don't I just use `center=False`? This causes a 1e-8 magnitude
    error in the output spectrogram, and a 1e-5 magnitude error in the
    MFCC, which is problematic for the PPG.
    """
    def forward(self, x):
        specgram = super().forward(x)[..., 1].transpose(-2, -1)
        return specgram