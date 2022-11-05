import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

from src.loss.auxiliary import AuxiliaryLoss
from src.data import DataProperties

################################################################################
# Frequency masking loss
################################################################################


class FrequencyMaskingLoss(AuxiliaryLoss):
    """
    Compute perceptually-inspired frequency masking loss. Given a time-aligned
    reference and additive perturbation, determine the degree to which the
    perturbation exceeds psychoacoustic thresholds induced by the reference (and
    thus the degree to which it is perceptible). Code adapted from
    https://bit.ly/3JjfSbI.
    """

    def __init__(self,
                 reduction: str = 'none',
                 window_size: int = 2048,
                 hop_size: int = 512,
                 normalize: str = None):
        super().__init__(reduction)

        self.sample_rate = DataProperties.get("sample_rate")
        self.scale = DataProperties.get("scale")

        self.window_size = window_size
        self.hop_size = hop_size

        # full-overlap: hop size must divide window size
        if self.window_size % self.hop_size:
            raise ValueError(f"Full-overlap: hop size {self.hop_size} must "
                             f"divide window size {self.window_size}")

        # encapsulate threshold computation
        self.masker = PsychoacousticMasker(
            window_size,
            hop_size,
            self.sample_rate
        )

        # normalize incoming audio to deal with loss scale-dependence
        if normalize not in [None, 'none', 'peak']:
            raise ValueError(f'Invalid normalization {normalize}')
        self.normalize = normalize

        # store reference thresholds and PSD maxima to avoid recomputing
        self.ref_wav = None
        self.ref_thresh = None
        self.ref_psd = None

    def _normalize(self, x: torch.Tensor):
        if self.normalize == "peak":
            (self.scale / torch.max(
                torch.abs(x) + 1e-8, dim=-1, keepdim=True
            )[0]) * x * 0.95
        elif self.normalize in [None, 'none']:
            pass
        else:
            raise ValueError(f'Invalid normalization {self.normalize}')
        return x

    def _pad(self, x: torch.Tensor):
        """
        Avoid boundary artifacts by padding inputs before STFT such that all
        samples are represented in the same number of spectrogram windows
        """
        pad_frames = self.window_size // self.hop_size - 1
        pad_len = pad_frames * self.hop_size
        return F.pad(x, (pad_len, pad_len))

    def _stabilized_threshold_and_psd_maximum(self, x_ref: torch.Tensor):
        """
        Return batch of stabilized masking thresholds and PSD maxima for
        reference audio

        :param x_ref: waveform reference inputs of shape (n_batch, ...)
        :return: tuple holding stabilized masking thresholds and PSD maxima
        """

        masking_threshold = []
        psd_maximum = []

        assert x_ref.ndim >= 2  # inputs must have batch dimension

        # apply padding to avoid boundary artifacts
        x_ref = self._pad(x_ref)

        for _, x_i in enumerate(x_ref):
            mt, pm = self.masker.calculate_threshold_and_psd_maximum(x_i)
            masking_threshold.append(mt)
            psd_maximum.append(pm)

        # stabilize loss by canceling out the "10*log" term in power spectral
        # density maximum and masking threshold
        masking_threshold_stabilized = 10 ** (
                torch.cat(masking_threshold, dim=0) * 0.1
        )
        psd_maximum_stabilized = 10 ** (torch.cat(psd_maximum, dim=0) * 0.1)

        return masking_threshold_stabilized, psd_maximum_stabilized

    def _masking_hinge_loss(
            self,
            perturbation: torch.Tensor,
            psd_maximum_stabilized: torch.Tensor,
            masking_threshold_stabilized: torch.Tensor
    ):
        """
        Compute hinge loss between aligned perturbation and frequency-masking
        thresholds induced by reference signal
        """

        n_batch = perturbation.shape[0]

        # calculate approximate power spectral density
        psd_perturbation = self._approximate_power_spectral_density(
            perturbation, psd_maximum_stabilized
        )

        # check that perturbation PSD is broadcastable with stored thresholds
        assert self._check_broadcastable(
            psd_perturbation, masking_threshold_stabilized
        ), f"Cannot broadcast perturbation PSD of shape " \
           f"{psd_perturbation.shape} with reference thresholds of shape " \
           f"{masking_threshold_stabilized.shape}"

        # calculate hinge loss per input, averaged over frames
        loss = nn.functional.relu(
            psd_perturbation - masking_threshold_stabilized
        ).view(n_batch, -1).mean(-1)

        return loss

    def _approximate_power_spectral_density(
            self,
            perturbation: torch.Tensor,
            psd_maximum_stabilized: torch.Tensor
    ):
        """
        Approximate power spectral density of perturbation
        """

        # require batch dimension
        assert perturbation.ndim >= 2

        n_batch = perturbation.shape[0]

        # pad to avoid boundary artifacts
        perturbation = self._pad(perturbation)

        # compute short-time Fourier transform (STFT)
        stft_matrix = torch.stft(
            perturbation.reshape(n_batch, -1),
            n_fft=self.window_size,
            hop_length=self.hop_size,
            win_length=self.window_size,
            center=False,
            return_complex=False,
            window=torch.hann_window(self.window_size).to(perturbation),
        ).to(perturbation)

        # compute power spectral density (PSD); fixes implementation of Qin et
        # al. by also considering the square root of gain_factor
        gain_factor = torch.sqrt(torch.as_tensor(8.0 / 3.0))
        psd_matrix = torch.sum(
            torch.square(gain_factor * stft_matrix / self.window_size),
            dim=-1
        )

        # approximate normalized psd using the following formula:
        # psd_matrix_approximated = 10^((96.0 - psd_matrix_max + psd_matrix)/10)
        psd_matrix_approximated = pow(
            10.0, 9.6
        ) / psd_maximum_stabilized.reshape(-1, 1, 1) * psd_matrix

        return psd_matrix_approximated  # (n_batch, n_freq, self.window_size)

    def _compute_loss(self, x: torch.Tensor, x_ref: torch.Tensor = None):
        """
        Compute frequency-masking loss.

        :param x: input, shape (n_batch, n_channels, signal_length)
        :param x_ref: reference, shape (n_batch, n_channels, signal_length) or
                      (1, n_channels, signal_length)
        :return: unreduced batch loss, shape (n_batch,)
        """

        # require batch dimension, discard channel dimension
        assert x.ndim >= 2
        n_batch, signal_length = x.shape[0], x.shape[-1]
        x = x.reshape(n_batch, signal_length)
        x = self._normalize(x)

        if x_ref is not None:
            assert x_ref.ndim >= 2
            x_ref = x_ref.reshape(x_ref.shape[0], x_ref.shape[-1])
            x_ref = self._normalize(x_ref)
            masking_threshold, psd_maximum = \
                self._stabilized_threshold_and_psd_maximum(x_ref)
        else:
            x_ref = self.ref_wav
            masking_threshold, psd_maximum = self.ref_thresh, self.ref_psd

        # check that waveform perturbation computation is broadcastable
        assert self._check_broadcastable(
            x, x_ref
        ), f"Cannot broadcast input waveform of shape {x.shape} " \
           f"with reference waveform of shape {x_ref.shape}"

        perturbation = x - x_ref  # assume additive waveform perturbation

        loss = self._masking_hinge_loss(
            perturbation, psd_maximum, masking_threshold
        )

        return loss

    def set_reference(self, x_ref: torch.Tensor):
        """
        Compute and store masking thresholds and PSD maxima for reference input

        :param x_ref: inputs, shape (n_batch, n_channels, signal_length)
        """

        # require batch dimension, discard channel dimension
        assert x_ref.ndim >= 2
        n_batch, signal_length = x_ref.shape[0], x_ref.shape[-1]
        x_ref = x_ref.clone().detach().reshape(n_batch, signal_length)
        self.ref_wav = self._normalize(x_ref)

        self.ref_thresh, self.ref_psd = \
            self._stabilized_threshold_and_psd_maximum(self.ref_wav)

        # do not track gradients for stored references
        self.ref_wav.requires_grad = False
        self.ref_thresh.requires_grad = False
        self.ref_psd.requires_grad = False


class PsychoacousticMasker:
    """
    Adapted from Adversarial Robustness Toolbox Imperceptible ASR attack.
    Implements psychoacoustic model of Lin and Abdulla (2015) following Qin et
    al. (2019) simplifications.

    | Repo link: https://github.com/Trusted-AI/adversarial-robustness-toolbox/
    | Paper link: Lin and Abdulla (2015), https://www.springer.com/gp/book/9783319079738
    | Paper link: Qin et al. (2019), http://proceedings.mlr.press/v97/qin19a.html
    """

    def __init__(self, window_size: int = 2048, hop_size: int = 512, sample_rate: int = 16000) -> None:
        """
        Initialization.

        :param window_size: Length of the window. The number of STFT rows is `(window_size // 2 + 1)`.
        :param hop_size: Number of audio samples between adjacent STFT columns.
        :param sample_rate: Sampling frequency of audio inputs.
        """
        self._window_size = window_size
        self._hop_size = hop_size
        self._sample_rate = sample_rate

        # init some private properties for lazy loading
        self._fft_frequencies = None
        self._bark = None
        self._absolute_threshold_hearing = None

    def calculate_threshold_and_psd_maximum(self,
                                            audio: torch.Tensor
                                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the global masking threshold for an audio input and also return
        its maximum power spectral density. This is the main method to call in
        order to obtain global masking thresholds for an audio input. It also
        returns the maximum power spectral density (PSD) for each frame. Given
        an audio input, the following steps are performed:

        1. STFT analysis and sound pressure level normalization
        2. Identification and filtering of maskers
        3. Calculation of individual masking thresholds
        4. Calculation of global masking thresholds

        :param audio: Audio samples of shape `(length,)`.
        :return: Global masking thresholds of shape
                 `(window_size // 2 + 1, frame_length)` and the PSD maximum for
                 each frame of shape `(frame_length)`.
        """

        assert audio.ndim <= 1 or audio.shape[0] == 1  # process a single waveform

        # compute normalized PSD estimate frame-by-frame for each input, as well
        # as maximum of each input's unnormalized PSD
        psd_matrix, psd_max = self.power_spectral_density(audio)
        threshold = torch.zeros_like(psd_matrix)

        # compute masking frequencies frame-by-frame for each input
        for frame in range(psd_matrix.shape[-1]):
            # apply methods for finding and filtering maskers
            maskers, masker_idx = self.filter_maskers(*self.find_maskers(psd_matrix[..., frame]))

            # apply methods for calculating global threshold
            threshold[..., frame] = self.calculate_global_threshold(
                self.calculate_individual_threshold(maskers, masker_idx)
            )

        return threshold, psd_max

    def power_spectral_density(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the power spectral density matrix for an audio input.

        :param audio: audio inputs of shape `(signal_len,)`.
        :return: PSD matrix of shape `(window_size // 2 + 1, frame_length)` and maximum vector of shape
        `(n_batch, frame_length)`.
        """

        # compute short-time Fourier transform (STFT)
        stft_matrix = torch.stft(
            audio.reshape(1, -1),
            n_fft=self.window_size,
            hop_length=self.hop_size,
            win_length=self.window_size,
            center=False,
            return_complex=True,
            window=torch.hann_window(self.window_size).to(audio.device),
        ).to(audio.device)

        # compute power spectral density (PSD)
        # note: fixes implementation of Qin et al. by also considering the square root of gain_factor
        gain_factor = torch.sqrt(torch.as_tensor(8.0 / 3.0))
        psd_matrix = 20 * torch.log10(torch.abs(gain_factor * stft_matrix / self.window_size))
        psd_matrix = psd_matrix.clamp(min=-200)

        # normalize PSD at 96dB
        psd_matrix_max = torch.amax(psd_matrix, dim=[d for d in range(1, psd_matrix.ndim)], keepdim=True)
        psd_matrix_normalized = 96.0 - psd_matrix_max + psd_matrix

        return psd_matrix_normalized, psd_matrix_max

    @staticmethod
    def find_maskers(psd_vector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Identify maskers. Possible maskers are local PSD maxima. Following Qin et al.,
        all maskers are treated as tonal.

        :param psd_vector: PSD vector of shape `(window_size // 2 + 1)`.
        :return: Possible PSD maskers and indices.
        """

        # find all local maxima in single-frame PSD estimate
        flat = psd_vector.reshape(-1)
        left = flat[1:-1] - flat[:-2]
        right = flat[1:-1] - flat[2:]

        ind = torch.where((left > 0) * (right > 0),
                          torch.ones_like(left),
                          torch.zeros_like(left))
        ind = torch.nn.functional.pad(ind, (1, 1), "constant", 0)
        masker_idx = torch.nonzero(ind, out=None).cpu().reshape(-1)

        # smooth maskers with their direct neighbors
        psd_maskers = 10 * torch.log10(
            torch.sum(
                torch.cat(
                    [10 ** (psd_vector[..., masker_idx + i] / 10) for i in range(-1, 2)]
                ),
                dim=0
            )
        )

        return psd_maskers, masker_idx

    def filter_maskers(self,
                       maskers: torch.Tensor,
                       masker_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Filter maskers. First, discard all maskers that are below the absolute threshold
        of hearing. Second, reduce pairs of maskers that are within 0.5 bark distance of
        each other by keeping the larger masker.

        :param maskers: Masker PSD values.
        :param masker_idx: Masker indices.
        :return: Filtered PSD maskers and indices.
        """
        # filter on the absolute threshold of hearing
        # note: deviates from Qin et al. implementation by filtering first on ATH and only then on bark distance
        ath_condition = maskers > self.absolute_threshold_hearing.to(maskers)[masker_idx]
        masker_idx = masker_idx[ath_condition]
        maskers = maskers[ath_condition]

        # filter on the bark distance
        bark_condition = torch.ones(masker_idx.shape, dtype=torch.bool)
        i_prev = 0
        for i in range(1, len(masker_idx)):
            # find pairs of maskers that are within 0.5 bark distance of each other
            if self.bark[i] - self.bark[i_prev] < 0.5:
                # discard the smaller masker
                i_todelete, i_prev = (i_prev, i_prev + 1) if maskers[i_prev] < maskers[i] else (i, i_prev)
                bark_condition[i_todelete] = False
            else:
                i_prev = i
        masker_idx = masker_idx[bark_condition]
        maskers = maskers[bark_condition]

        return maskers, masker_idx

    @property
    def window_size(self) -> int:
        """
        :return: Window size of the masker.
        """
        return self._window_size

    @property
    def hop_size(self) -> int:
        """
        :return: Hop size of the masker.
        """
        return self._hop_size

    @property
    def sample_rate(self) -> int:
        """
        :return: Sample rate of the masker.
        """
        return self._sample_rate

    @property
    def fft_frequencies(self) -> torch.Tensor:
        """
        :return: Discrete fourier transform sample frequencies.
        """
        if self._fft_frequencies is None:
            self._fft_frequencies = torch.linspace(0, self.sample_rate / 2, self.window_size // 2 + 1)
        return self._fft_frequencies

    @property
    def bark(self) -> torch.Tensor:
        """
        :return: Bark scale for discrete fourier transform sample frequencies.
        """
        if self._bark is None:
            self._bark = 13 * torch.arctan(0.00076 * self.fft_frequencies) + 3.5 * torch.arctan(
                torch.square(self.fft_frequencies / 7500.0)
            )
        return self._bark

    @property
    def absolute_threshold_hearing(self) -> torch.Tensor:
        """
        :return: Absolute threshold of hearing (ATH) for discrete fourier transform sample frequencies.
        """
        if self._absolute_threshold_hearing is None:
            # ATH applies only to frequency range 20Hz<=f<=20kHz
            # note: deviates from Qin et al. implementation by using the Hz range as valid domain
            valid_domain = torch.logical_and(20 <= self.fft_frequencies, self.fft_frequencies <= 2e4)
            freq = self.fft_frequencies[valid_domain] * 0.001

            # outside valid ATH domain, set values to -infinity
            # note: This ensures that every possible masker in the bins <=20Hz is valid. As a consequence, the global
            # masking threshold formula will always return a value different to infinity
            self._absolute_threshold_hearing = torch.ones(valid_domain.shape) * -float('inf')

            self._absolute_threshold_hearing[valid_domain] = (
                    3.64 * pow(freq, -0.8) - 6.5 * torch.exp(-0.6 * torch.square(freq - 3.3)) + 0.001 * pow(freq, 4) - 12
            )
        return self._absolute_threshold_hearing

    def calculate_individual_threshold(self,
                                       maskers: torch.Tensor,
                                       masker_idx: torch.Tensor) -> torch.Tensor:
        """
        Calculate individual masking threshold with frequency denoted at bark scale.

        :param maskers: Masker PSD values.
        :param masker_idx: Masker indices.
        :return: Individual threshold vector of shape `(window_size // 2 + 1)`.
        """
        delta_shift = -6.025 - 0.275 * self.bark
        threshold = torch.zeros(masker_idx.shape + self.bark.shape).to(maskers)

        for k, (masker_j, masker) in enumerate(zip(masker_idx, maskers)):

            # critical band rate of the masker
            z_j = self.bark[masker_j].to(maskers)
            # distance maskees to masker in bark
            delta_z = self.bark.to(maskers) - z_j

            # define two-slope spread function:
            #   if delta_z <= 0, spread_function = 27*delta_z
            #   if delta_z > 0, spread_function = [-27+0.37*max(PSD_masker-40,0]*delta_z
            spread_function = 27 * delta_z
            spread_function[delta_z > 0] = (-27 + 0.37 * max(masker - 40, 0)) * delta_z[delta_z > 0]

            # calculate threshold
            threshold[k, :] = masker + delta_shift[masker_j] + spread_function

        return threshold

    def calculate_global_threshold(self, individual_threshold):
        """
        Calculate global masking threshold.

        :param individual_threshold: Individual masking threshold vector.
        :return: Global threshold vector of shape `(window_size // 2 + 1)`.
        """
        # note: deviates from Qin et al. implementation by taking the log of the summation, which they do for numerical
        #       stability of the stage 2 optimization. We stabilize the optimization in the loss itself.

        return 10 * torch.log10(
            torch.sum(10 ** (individual_threshold / 10), dim=0) + 10 ** (self.absolute_threshold_hearing.to(individual_threshold) / 10)
        )
