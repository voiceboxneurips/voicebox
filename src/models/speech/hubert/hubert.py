import torch
import torchaudio

import torch.nn as nn

from src.data import DataProperties

torchaudio.set_audio_backend("sox_io")


class HUBERT(nn.Module):

    def __init__(self, variant: str = "large"):

        super().__init__()

        # identify model variant (distinguished by size, dataset, fine-tuning)
        variants = {

            # `LARGE` variant, trained on Libri-Light 60,000h, fine-tuned on
            # full LibriSpeech 960h
            "large": "HUBERT_ASR_LARGE",

            # `XLARGE` variant, trained on Libri-Light 60,000h, fine-tuned on
            # full LibriSpeech 960h
            "xlarge": "HUBERT_ASR_XLARGE"

        }

        try:
            variant_full_name = variants[variant]
        except KeyError:
            raise ValueError(f"Invalid variant {variant}; must be one of "
                             f"{list(variants.keys())}")

        # import HUBERT model variant as `bundle` object
        bundle = eval(f'torchaudio.pipelines.{variant_full_name}')

        # unpack model, labels, and sample rate
        self.model = bundle.get_model()
        self.labels = bundle.get_labels()
        self.sample_rate = bundle.sample_rate

        # hardcode for HUBERT
        self.sep_idx = 4
        self.blank_idx = 0

        # check sample rate
        if DataProperties.get("sample_rate") != self.sample_rate:
            raise ValueError(f"Incompatible data and model sample rates "
                             f"{DataProperties.get('sample_rate')}, "
                             f"{self.sample_rate}")

        # feature extractor: stacked 1D convolutional blocks
        assert self.model.feature_extractor is not None

        # encoder: feature projection, transformer
        assert self.model.encoder is not None

        # aux: fine-tuned linear layer(s) mapping to token probabilities
        assert self.model.aux is not None

    def forward(self, x: torch.Tensor):
        """
        Pass input audio through feature extractor, encoder, and fine-tuned
        auxiliary layer to produce a sequence of token probability distributions

        :param x: waveform audio of shape (n_batch, ..., signal_length)
        :return:
        """

        # reshape audio to (n_batch, signal_length)
        if x.ndim != 2:
            n_batch, signal_length = x.shape[0], x.shape[-1]
            x = x.reshape(n_batch, signal_length)

        # emit sequence(s) of token probabilities
        emission, _ = self.model(x, lengths=None)

        return emission

    def extract_features(self, x: torch.Tensor):
        """
        Extract deep features.
        """

        # reshape audio to (n_batch, signal_length)
        if x.ndim != 2:
            n_batch, signal_length = x.shape[0], x.shape[-1]
            x = x.reshape(n_batch, signal_length)

        return self.model.extract_features(x)[0]
