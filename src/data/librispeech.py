import os
import math

import librosa as li
import numpy as np
import textgrid

import torch


from src.data import DataProperties, VoiceBoxDataset
from src.utils import ensure_dir
from src.constants import (
    LIBRISPEECH_DATA_DIR,
    LIBRISPEECH_CACHE_DIR,
    SAMPLE_RATE,
    LIBRISPEECH_EXT,
    LIBRISPEECH_PHONEME_EXT,
    LIBRISPEECH_PHONEME_DICT,
    LIBRISPEECH_SIG_LEN,
    HOP_LENGTH
)
from src.attacks.offline.perturbation.voicebox.voicebox import PitchEncoder

from os import path
from tqdm import tqdm
from pathlib import Path

from typing import Union, Iterable


################################################################################
# Cache and load LibriSpeech dataset
################################################################################


class LibriSpeechDataset(VoiceBoxDataset):
    """
    A Dataset object for the LibriSpeech dataset subsets. The required data can
    be downloaded by running the script `download_librispeech.sh`. This class
    takes audio data from the specified directory and caches tensors to disk.
    """
    def __init__(self,
                 split: str = 'test-clean',
                 data_dir: str = LIBRISPEECH_DATA_DIR,
                 cache_dir: str = LIBRISPEECH_CACHE_DIR,
                 sample_rate: int = SAMPLE_RATE,
                 audio_ext: str = LIBRISPEECH_EXT,
                 phoneme_ext: str = LIBRISPEECH_PHONEME_EXT,
                 signal_length: Union[float, int] = LIBRISPEECH_SIG_LEN,
                 scale: Union[float, int] = 1.0,
                 hop_length: int = HOP_LENGTH,
                 target: str = 'speaker',
                 features: Union[str, Iterable[str]] = None,
                 batch_format: str = 'dict',
                 *args,
                 **kwargs):
        """
        Load, organize, and cache LibriSpeech dataset.

        Parameters
        ----------
        split (str):

        data_dir (str):      LibriSpeech root directory

        cache_dir (str):     root directory to which tensors will be saved

        sample_rate (int):   sample rate in Hz

        audio_ext (str):     extension for audio files within dataset

        phoneme_ext (str):   extension for phoneme alignment files within
                             dataset

        signal_length (int): length of audio files in samples (if `int` given)
                             or seconds (if `float` given)

        scale (float):       range to which audio will be scaled

        hop_length (int):    hop size for computing frame-wise features (e.g.
                             pitch, loudness)

        target (str):        string specifying target type. Must be one of
                             `speaker` (speaker ID), `phoneme` (aligned phoneme
                             labels), or `transcript`

        features (Iterable): strings specifying features to compute for each
                             audio file in the dataset. Must be subset of
                             `pitch`, `periodicity`, `loudness`

        batch_format (str):  format for returning batches. Must be either `dict`
                             or `tuple`
        """

        self.phoneme_ext = phoneme_ext
        self.phoneme_list = []

        super().__init__(
            split=split,
            data_dir=data_dir,
            cache_dir=cache_dir,
            audio_ext=audio_ext,
            signal_length=signal_length,
            scale=scale,
            target=target,
            features=features,
            sample_rate=sample_rate,
            hop_length=hop_length,
            batch_format=batch_format,
            *args, **kwargs
        )

    def __str__(self):
        """Return string representation of dataset"""
        return f'LibriSpeechDataset(split={self.split}, ' \
               f'target={self.target}, features={self.features})'

    @staticmethod
    def _check_split(split: str):
        """Check for valid dataset split"""
        if split not in [
            'test-clean',
            'test-other',
            'dev-clean',
            'dev-other',
            'train-clean-100',
            'train-clean-360',
            'train-other-500'
        ]:
            raise ValueError(f'Invalid split {split}')
        return split

    @staticmethod
    def _check_target(target: str):
        if target not in ['speaker', 'phoneme', 'transcript']:
            raise ValueError(f'Invalid target type {target}')
        return target

    def _get_target_id(self):
        """Identifier for cached targets"""
        if self.target in ['speaker', 'transcript']:
            return f'{self.target}'
        else:
            return f'{self.sample_rate}-{self.hop_length}-{self.target}'

    def _get_audio_list(self, *args, **kwargs):
        """
        Scan for all audio files with given extension. Additionally, only select
        audio files for which corresponding phoneme alignments exist.
        """

        audio_files = [os.path.splitext(f)[0] for f in
                       (Path(self.data_dir) / self.split).rglob(
                           f'*.{self.audio_ext}')]
        phoneme_files = [os.path.splitext(f)[0] for f in
                         (Path(self.data_dir) / self.split).rglob(
                             f'*.{self.phoneme_ext}')]
        matching_files = list(set(audio_files) & set(phoneme_files))

        return sorted(
            [f + "." + self.audio_ext for f in matching_files]
        )

    def _build_target_cache(self):
        """Process and cache targets"""

        target_id = self._get_target_id()
        target_cache = list(
            (Path(self.cache_dir) / self.split).rglob(
                f'{target_id}.pt')
        )
        if len(target_cache) >= 1:
            return

        # speaker ID targets
        if self.target == 'speaker':

            targets = torch.zeros(
                len(self.audio_list), dtype=torch.long
            )

            pbar = tqdm(self.audio_list, total=len(self.audio_list))
            for i, audio_fn in enumerate(pbar):
                pbar.set_description(
                    f'Loading Speaker IDs ({self.split}): '
                    f'{path.basename(audio_fn)}')

                # extract speaker ID
                targets[i] = int(Path(audio_fn).parts[-3])

        # frame-aligned phoneme label targets
        elif self.target == 'phoneme':

            # retrieve phoneme alignment files
            self.phoneme_list = [
                os.path.splitext(f)[0] +
                "." + self.phoneme_ext for f in self.audio_list]

            targets = torch.zeros(len(self.phoneme_list),
                                  self.num_frames,
                                  dtype=torch.long)

            pbar = tqdm(self.phoneme_list, total=len(self.phoneme_list))
            for i, phoneme_fn in enumerate(pbar):

                pbar.set_description(
                    f'Loading phoneme alignments ({self.split}): '
                    f'{path.basename(phoneme_fn)}')

                # load interval labels from TextGrid format
                tg = textgrid.TextGrid.fromFile(phoneme_fn)
                if tg[0].name == 'phones':
                    phoneme_intervals = tg[0]
                elif tg[1].name == 'phones':
                    phoneme_intervals = tg[1]
                else:
                    raise ValueError("Could not find phonemes")

                # compute number of frames in audio file given hop size,
                # rounding up
                num_frames = math.ceil(
                    tg.maxTime * self.sample_rate / self.hop_length)
                ppg = torch.zeros(num_frames, dtype=torch.long)

                # for each labeled interval, break up into frames with given hop
                # size and assign phoneme labels
                for interval in phoneme_intervals:
                    interval.minTime = math.ceil(
                        interval.minTime * self.sample_rate / self.hop_length)
                    interval.maxTime = math.ceil(
                        interval.maxTime * self.sample_rate / self.hop_length)
                    phoneme_idx = LIBRISPEECH_PHONEME_DICT[interval.mark]
                    ppg[interval.minTime:interval.maxTime+1] = phoneme_idx

                targets[
                    i, :min(ppg.shape[-1], self.num_frames)
                ] = ppg[..., :self.num_frames]

        # string transcript targets
        elif self.target == 'transcript':
            raise NotImplementedError()

        else:
            raise ValueError(f'Invalid target type {self.target}')

        # cache targets to disk
        torch.save(targets,
                   path.join(
                       self.cache_dir,
                       self.split,
                       f'{target_id}.pt'
                   ))
