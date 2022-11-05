import torch
import torch.nn as nn
import torch.nn.functional as F

import math

import librosa as li

from typing import Union

from src.attacks.offline.trainable import TrainableAttack
from src.pipelines import Pipeline
from src.loss.adversarial import AdversarialLoss
from src.attacks.offline.perturbation import AdditivePerturbation
from src.data import DataProperties

################################################################################
# Implementation of universal additive attack of Li et al.
################################################################################


class AdvPulseAttack(TrainableAttack):

    def __init__(self,
                 pipeline: Pipeline,
                 adv_loss: AdversarialLoss,
                 mimic_sound: Union[torch.Tensor, str] = None,
                 init_mimic: bool = False,
                 tile_reference: bool = False,
                 normalize: bool = False,
                 eps: float = 0.05,
                 pgd_norm: Union[str, int, float] = float('inf'),
                 length: Union[int, float] = 0.5,
                 align: str = 'start',
                 loop: bool = False,
                 **kwargs
                 ):

        super().__init__(
            pipeline=pipeline,
            adv_loss=adv_loss,
            perturbation=AdditivePerturbation(
                eps=eps,
                projection_norm=pgd_norm,
                length=length,
                align=align,
                loop=loop,
                normalize=normalize
            ),
            **kwargs
        )

        # determine whether to repeat template to match perturbation length
        self.tile_reference = tile_reference

        if mimic_sound is None:
            self.mimic_sound = None

        elif isinstance(mimic_sound, torch.Tensor):

            # require batch, channel dimensions
            assert mimic_sound.ndim >= 2

            # convert to mono audio
            if mimic_sound.ndim == 2:
                mimic_sound = mimic_sound.unsqueeze(1)
            self.mimic_sound = mimic_sound.mean(
                dim=1, keepdim=True
            ).to(self.pipeline.device)

        # load from file path
        elif isinstance(mimic_sound, str):

            # load from randomly-selected file
            mimic_sound_np, _ = li.load(
                mimic_sound,
                sr=DataProperties.get('sample_rate'),
                mono=True
            )
            mimic_sound = torch.as_tensor(mimic_sound_np)

            # if length is specified, trim to match
            max_len = DataProperties.get('signal_length')
            self.mimic_sound = mimic_sound[..., :max_len].reshape(
                1, 1, -1
            ).to(self.pipeline.device)

        else:
            raise ValueError(f'Invalid mimic sound type {type(mimic_sound)}')

        # if specified, initialize adversarial perturbation to match template
        if self.mimic_sound is not None and init_mimic:
            self.perturbation.delta = nn.Parameter(
                self._match_signal_length(
                    self.mimic_sound,
                    torch.zeros(1, self.perturbation.length)
                )
            )

    @staticmethod
    def _crossfade(sig, fade_len):
        """Apply cross-fade to ends of signal"""

        sig = sig.clone()
        fade_len = int(fade_len * sig.shape[-1])
        fade_in = torch.linspace(0, 1, fade_len).to(sig)
        fade_out = torch.linspace(1, 0, fade_len).to(sig)
        sig[..., :fade_len] *= fade_in
        sig[..., -fade_len:] *= fade_out
        return sig

    def _match_signal_length(self, sig: torch.Tensor, ref: torch.Tensor):
        """
        Match length of signal to reference, either by trimming or repeating and
        cross-fading
        """

        sig = sig.reshape(1, -1)
        ref = ref.reshape(1, -1)

        signal_length = ref.shape[-1]
        if sig.shape[-1] >= signal_length:
            return sig[..., :signal_length].reshape(1, 1, -1).to(ref)
        elif not self.tile_reference:
            return F.pad(
                sig, (0, signal_length - sig.shape[-1])
            ).reshape(1, 1, -1).to(ref)

        # cross-fade length
        overlap = 0.05

        step = math.ceil(sig.shape[-1] * (1 - overlap))
        n_repeat = math.ceil(signal_length / step)

        padded = torch.zeros(
            1, step * (n_repeat - 1) + sig.shape[-1] + 1
        ).reshape(1, -1).to(sig)
        shape = padded.shape[:-1] + (n_repeat, sig.shape[-1])

        strides = (padded.stride()[0],) + (step, padded.stride()[-1],)
        frames = torch.as_strided(
            padded, size=shape, stride=strides
        )[::step]

        for j in range(n_repeat):
            frames[:, j, :] += self._crossfade(sig, overlap)

        sig = padded[..., :signal_length].reshape(
            1, 1, -1
        ).to(ref)

        return sig

    def _set_loss_reference(self, x: torch.Tensor):
        """
        Pass reference audio to auxiliary loss to avoid re-computing expensive
        intermediate representations. For AdvPulse attack, optionally use
        """

        if self.aux_loss is not None:

            if self.mimic_sound is not None:
                reference = self._match_signal_length(
                    self.mimic_sound,
                    self.perturbation.delta
                )
            else:
                reference = x

            self.aux_loss.set_reference(reference)

    def _compute_aux_loss(self,
                          x_adv: torch.Tensor,
                          x_ref: torch.Tensor = None):
        """Compute auxiliary loss, optionally """
        if self.mimic_sound is not None:
            return self.aux_loss(self.perturbation.delta, x_ref)
        else:
            return self.aux_loss(x_adv, x_ref)

    def _log_step(self,
                  x: torch.Tensor,
                  x_adv: torch.Tensor,
                  y: torch.Tensor,
                  adv_loss: Union[float, torch.Tensor] = None,
                  det_loss: Union[float, torch.Tensor] = None,
                  aux_loss: Union[float, torch.Tensor] = None,
                  success_rate: Union[float, torch.Tensor] = None,
                  detection_rate: Union[float, torch.Tensor] = None,
                  idx: int = 0,
                  tag: str = None,
                  *args,
                  **kwargs
                  ):

        if self.writer is None or self._iter_id % self.writer.log_iter:
            return

        if tag is None:
            tag = f'{self.__class__.__name__}-' \
                  f'{self.aux_loss.__class__.__name__}'

        super()._log_step(
            x,
            x_adv,
            y,
            adv_loss=adv_loss,
            det_loss=det_loss,
            aux_loss=aux_loss,
            success_rate=success_rate,
            detection_rate=detection_rate,
            idx=idx,
            tag=tag
        )

        # add audio and spectrogram logging for mimic sound
        if self.mimic_sound is not None:
            self.writer.log_audio(
                self.mimic_sound,
                f'{tag}/sound-template',
                global_step=self._iter_id
            )
