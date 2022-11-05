import os
import math
from copy import deepcopy

import librosa as li
import numpy as np

import torch
import torch.nn.functional as F

from torch.utils.data import Dataset

from src.data.dataproperties import DataProperties
from src.constants import (
    SAMPLE_RATE,
    HOP_LENGTH
)
from src.attacks.offline.perturbation.voicebox.pitch import PitchEncoder
from src.attacks.offline.perturbation.voicebox.loudness import LoudnessEncoder

from os import path
from tqdm import tqdm
from pathlib import Path

from typing import Union, Iterable

################################################################################
# Cache and load datasets
################################################################################


def ensure_dir(directory: Union[str, Path]):
    """
    Ensure all directories along given path exist, given directory name
    """
    directory = str(directory)
    if len(directory) > 0 and not os.path.exists(directory):
        os.makedirs(directory)


class VoiceBoxDataset(Dataset):

    """
    A Dataset object for the LibriSpeech dataset subsets. The required data can
    be downloaded by running the script `download_librispeech.sh`. This class
    takes audio data from the specified directory and caches tensors to disk.
    """
    def __init__(self,
                 split: str,
                 data_dir: str,
                 cache_dir: str,
                 audio_ext: str,
                 signal_length: Union[float, int],
                 scale: Union[float, int],
                 target: str,
                 features: Union[str, Iterable[str]] = None,
                 sample_rate: int = SAMPLE_RATE,
                 hop_length: int = HOP_LENGTH,
                 batch_format: str = 'dict',
                 *args,
                 **kwargs):
        """
        Load, organize, and cache LibriSpeech dataset.

        Parameters
        ----------
        split (str):         data subset name

        data_dir (str):      dataset root directory

        cache_dir (str):     root directory to which tensors will be saved

        sample_rate (int):   sample rate in Hz

        audio_ext (str):     extension for audio files within dataset

        signal_length (int): length of audio files in samples (if `int` given)
                             or seconds (if `float` given)

        scale (float):       range to which audio will be scaled

        hop_length (int):    hop size for computing frame-wise features (e.g.
                             pitch, loudness)

        target (str):        string specifying target type.

        features (Iterable): strings specifying features to compute for each
                             audio file in the dataset. Must be subset of
                             `pitch`, `periodicity`, `loudness`

        batch_format (str):  format for returning batches. Must be either `dict`
                             or `tuple`
        """

        if batch_format not in ['dict', 'tuple']:
            raise ValueError(f'Invalid batch format {batch_format}')
        self.batch_format = batch_format

        self.data_dir = os.fspath(data_dir)
        self.cache_dir = os.fspath(cache_dir)

        self.audio_ext = audio_ext
        self.sample_rate = sample_rate
        self.scale = scale
        self.hop_length = hop_length

        # if signal length is given as floating-point value, assume time in
        # seconds and convert to samples
        if isinstance(signal_length, float):
            self.signal_length = math.floor(signal_length * self.sample_rate)
        else:
            self.signal_length = signal_length

        # compute frame-equivalent signal length for targets/features,
        # accounting for center-padding in spectrogram implementations
        self.num_frames = math.ceil(self.signal_length / self.hop_length)
        if not self.signal_length % self.hop_length:
            self.num_frames += 1

        # register data properties
        DataProperties.register_properties(
            sample_rate=self.sample_rate,
            signal_length=self.signal_length,
            scale=self.scale
        )

        # check for valid subset
        self.split = self._check_split(split)

        # create directories if necessary
        ensure_dir(path.join(self.cache_dir, self.split))
        ensure_dir(path.join(self.cache_dir, self.split))

        # check for valid target types
        self.target = self._check_target(target)

        # check for valid feature types
        self.features = self._check_features(features)

        # scan all audio files in dataset
        self.audio_list = self._get_audio_list()

        # check for cached audio, targets, and features by name. If missing,
        # build required caches. Cache files are identified by sample rate and
        # hop size where necessary (e.g. for pitch features, but not class
        # targets)
        self._build_audio_cache()
        self._build_target_cache()
        for feature in self.features:
            self._build_feature_cache(feature)

        # load data and target tensors from caches
        self.tx = torch.load(
            Path(self.cache_dir) /
            self.split /
            f'{self._get_audio_id()}.pt')
        self.ty = torch.load(
            Path(self.cache_dir) /
            self.split /
            f'{self._get_target_id()}.pt')

        # load feature tensors from cache and store by name
        self.tf = dict()
        if self.features is not None and self.features:
            for feature in self.features:
                self.tf[feature] = torch.load(
                    Path(self.cache_dir) /
                    self.split /
                    f'{self._get_feature_id(feature)}.pt')

    @staticmethod
    def _check_split(split: str):
        if split not in ['train', 'test']:
            raise ValueError(f'Invalid split {split}')
        return split

    @staticmethod
    def _check_target(target: str):
        if target not in ['class', 'transcript']:
            raise ValueError(f'Invalid target type {target}')
        return target

    @staticmethod
    def _check_features(features: Union[str, Iterable[str]]):
        if features is None or not features:
            features = []
        else:
            if isinstance(features, str):
                features = [features]

            for f in features:
                if f not in ['pitch', 'periodicity', 'loudness']:
                    raise ValueError(f'Invalid feature type {f}')
        return list(features)

    def _get_audio_list(self, *args, **kwargs):
        """Scan for all audio files with given extension"""
        return sorted(
            list((Path(self.data_dir) / self.split).rglob(
                f'*.{self.audio_ext}'))
        )

    def _get_audio_id(self):
        """Identifier for cached audio"""
        return f'{self.sample_rate}-audio'

    def _get_target_id(self):
        """Identifier for cached targets"""
        if self.target in ['class', 'transcript']:
            return f'{self.target}'
        else:
            return f'{self.sample_rate}-{self.hop_length}-{self.target}'

    def _get_feature_id(self, feature: str):
        """Identifier for cached features"""
        return f'{self.sample_rate}-{self.hop_length}-{feature}'

    def _build_audio_cache(self):
        """Load audio data and cache to disk"""

        audio_id = self._get_audio_id()
        audio_cache = list(
            (Path(self.cache_dir) / self.split).rglob(
                f'{audio_id}.pt')
        )
        if len(audio_cache) >= 1:
            return

        # prepare to store audio waveforms and lengths
        waveforms = torch.zeros(len(self.audio_list), 1, self.signal_length)

        pbar = tqdm(self.audio_list, total=len(self.audio_list))
        for i, audio_fn in enumerate(pbar):
            pbar.set_description(
                f'Loading {self.split}: {path.basename(audio_fn)}')

            # load audio and resample, but leave original length
            waveform, _ = li.load(audio_fn,
                                  mono=True,
                                  sr=self.sample_rate)
            waveforms[
                i, :, :min(self.signal_length, len(waveform))
            ] = torch.from_numpy(waveform)[..., :self.signal_length]

        # cache padded tensors and lengths to disk
        torch.save(waveforms,
                   path.join(
                       self.cache_dir,
                       self.split,
                       f'{audio_id}.pt')
                   )

    def _build_target_cache(self):
        """Load targets and cache to disk"""
        raise NotImplementedError()

    def _build_feature_cache(self, feature: str):
        """Load features and cache to disk"""

        feature_id = self._get_feature_id(feature)
        feature_cache = list(
            (Path(self.cache_dir) / self.split).rglob(
                f'{feature_id}.pt')
        )
        if len(feature_cache) >= 1:
            return

        # compute f0, periodicity using PyWorld 'dio' algorithm
        pitch_extractor = PitchEncoder(hop_length=self.hop_length)
        loudness_extractor = LoudnessEncoder(hop_length=self.hop_length)

        # determine 'zero' values for each feature
        zero_pitch, zero_per = pitch_extractor(
            torch.zeros(1, 1, self.signal_length))
        zero_loud = loudness_extractor(torch.zeros(1, 1, self.signal_length))
        pad_val_pitch = zero_pitch.mean().item()
        pad_val_per = zero_per.mean().item()
        pad_val_loud = zero_loud.mean().item()

        # store frame-wise features
        if feature == 'loudness':
            loudness = torch.full(
                (len(self.audio_list), self.num_frames, 1),
                pad_val_loud,
                dtype=torch.float32
            )
        elif feature in ['pitch', 'periodicity']:
            pitch = torch.full(
                (len(self.audio_list), self.num_frames, 1),
                pad_val_pitch,
                dtype=torch.float32
            )
            periodicity = torch.full(
                (len(self.audio_list), self.num_frames, 1),
                pad_val_per,
                dtype=torch.float32
            )

        # iterate over audio
        pbar = tqdm(self.audio_list, total=len(self.audio_list))
        for i, audio_fn in enumerate(pbar):
            pbar.set_description(
                f'Computing {feature} ({self.split}): '
                f'{path.basename(audio_fn)}')

            # load audio and resample, but leave original length
            waveform, _ = li.load(audio_fn,
                                  mono=True,
                                  sr=self.sample_rate,
                                  duration=self.signal_length / self.sample_rate)

            # convert to tensor, insert batch dimension
            waveform = torch.from_numpy(waveform).unsqueeze(0)

            # trim or pad waveform if necessary
            if waveform.shape[-1] >= self.signal_length:
                waveform = waveform[..., :self.signal_length]
            else:
                pad_len = self.signal_length - waveform.shape[-1]
                waveform = F.pad(waveform, (0, pad_len))

            # compute and store pitch/periodicity in tandem
            if feature in ['pitch', 'periodicity']:

                f0, p = pitch_extractor(waveform)
                pitch[
                    i, :min(f0.shape[1], self.num_frames), :
                ] = f0[:, :self.num_frames, :]
                periodicity[
                    i, :min(p.shape[1], self.num_frames), :
                ] = p[:, :self.num_frames, :]

            elif feature == 'loudness':

                l = loudness_extractor(waveform)
                loudness[
                    i, :min(l.shape[1], self.num_frames), :
                ] = l[:, :self.num_frames, :]

            else:
                raise ValueError(f'Invalid feature type {feature}')

        if feature in ['pitch', 'periodicity']:

            # save to disk
            torch.save(pitch,
                       path.join(
                           self.cache_dir,
                           self.split,
                           f'{self._get_feature_id("pitch")}.pt'
                       ))
            torch.save(periodicity,
                       path.join(
                           self.cache_dir,
                           self.split,
                           f'{self._get_feature_id("periodicity")}.pt'
                       ))
        else:
            # save to disk
            torch.save(loudness,
                       path.join(
                           self.cache_dir,
                           self.split,
                           f'{feature_id}.pt'
                       ))

    def __len__(self):
        return len(self.tx)

    def __getitem__(self, idx):
        """Return batch of audio, targets, and optional feature values"""

        if self.batch_format == 'dict':
            # return batch items by name
            batch = {
                'x': self.tx[idx],
                'y': self.ty[idx],
                **{k: self.tf[k][idx] for k in self.tf}
            }
        elif self.batch_format == 'tuple':
            # return batch items in order
            batch = (self.tx[idx], self.ty[idx]) + tuple(
                self.tf[k][idx] for k in self.tf)
        else:
            raise ValueError(f'Invalid batch format {self.batch_format}')

        return batch

    def index_reduce(self, idx):
        """Reduce to a subset by indexing into all stored tensors"""

        new_dataset = deepcopy(self)
        new_dataset.tx = new_dataset.tx[idx]
        new_dataset.ty = new_dataset.ty[idx]
        for feature in new_dataset.features:
            new_dataset.tf[feature] = new_dataset.tf[feature][idx]

        return new_dataset

    def overwrite_dataset(self, x: torch.Tensor, y: torch.Tensor, idx):
        """Overwrite inputs and targets, and select features correspondingly"""

        # support boolean or integer indices
        assert len(idx) <= self.__len__()
        assert len(idx) == self.__len__() or \
               (len(idx) == len(x) and len(idx) == len(y))

        new_dataset = self.index_reduce(idx)
        new_dataset.tx = x
        new_dataset.ty = y

        return new_dataset
