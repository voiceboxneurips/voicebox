import torch
import torchaudio
import torch.nn as nn

from src.data import DataProperties

torchaudio.set_audio_backend("sox_io")


class Wav2Vec2(nn.Module):
    """
    Wav2Vec2 ASR model, as proposed by Baevski et al.
    (https://arxiv.org/abs/2006.11477). Takes arbitrary-length waveform audio
    at 16kHz and produces string transcripts
    """

    def __init__(self, variant: str = "base_960h"):

        super().__init__()

        # identify model variant (distinguished by size, dataset, fine-tuning)
        variants = {

            # `BASE` variant, trained on LibriSpeech 960h, fine-tuned on 10
            # minutes of Libri-Light
            "base_10m": "WAV2VEC2_ASR_BASE_10M",

            # `BASE` variant, trained on LibriSpeech 960h, fine-tuned on 100
            # hours of LibriSpeech `train-clean-100` subset
            "base_100h": "WAV2VEC2_ASR_BASE_100H",

            # `BASE` variant, trained on LibriSpeech 960h, fine-tuned on full
            # LibriSpeech 960h
            "base_960h": "WAV2VEC2_ASR_BASE_960H",

            # `LARGE` variant, trained on LibriSpeech 960h, fine-tuned on 10
            # minutes of Libri-Light
            "large_10m": "WAV2VEC2_ASR_LARGE_10M",

            # `LARGE` variant, trained on LibriSpeech 960h, fine-tuned on 100
            # hours of LibriSpeech `train-clean-100` subset
            "large_100h": "WAV2VEC2_ASR_LARGE_100H",

            # `LARGE` variant, trained on LibriSpeech 960h, fine-tuned on full
            # LibriSpeech 960h
            "large_960h": "WAV2VEC2_ASR_LARGE_960H",

            # `LARGE` variant, trained on Libri-Light 60,000h, fine-tuned on 10
            # minutes of Libri-Light
            "large_lv60k_10m": "WAV2VEC2_ASR_LARGE_LV60K_10M",

            # `LARGE` variant, trained on Libri-Light 60,000h, fine-tuned on 100
            # hours of LibriSpeech `train-clean-100` subset
            "large_lv60k_100h": "WAV2VEC2_ASR_LARGE_LV60K_100H",

            # `LARGE` variant, trained on Libri-Light 60,000h, fine-tuned on full
            # LibriSpeech 960h
            "large_lv60k_960h": "WAV2VEC2_ASR_LARGE_LV60K_960H",

            # `BASE` variant, trained on VoxPopuli 10,000h, fine-tuned on 282h
            # German subset
            "base_10k_de": "VOXPOPULI_ASR_BASE_10K_DE",

            # `BASE` variant, trained on VoxPopuli 10,000h, fine-tuned on 543h
            # English subset
            "base_10k_en": "VOXPOPULI_ASR_BASE_10K_EN",

            # `BASE` variant, trained on VoxPopuli 10,000h, fine-tuned on 166h
            # Spanish subset
            "base_10k_es": "VOXPOPULI_ASR_BASE_10K_ES",

            # `BASE` variant, trained on VoxPopuli 10,000h, fine-tuned on 211h
            # French subset
            "base_10k_fr": "VOXPOPULI_ASR_BASE_10K_FR",

            # `BASE` variant, trained on VoxPopuli 10,000h, fine-tuned on 91h
            # Spanish subset
            "base_10k_it": "VOXPOPULI_ASR_BASE_10K_IT"
        }

        try:
            variant_full_name = variants[variant]
        except KeyError:
            raise ValueError(f"Invalid variant {variant}; must be one of "
                             f"{list(variants.keys())}")

        # import Wav2Vec2 model variant as `bundle` object
        bundle = eval(f'torchaudio.pipelines.{variant_full_name}')

        # unpack model, labels, and sample rate
        self.model = bundle.get_model()
        self.labels = bundle.get_labels()
        self.sample_rate = bundle.sample_rate

        # hardcode for Wav2Vec2
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
