import io
import torch
import torch.nn.functional as F
import math
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import IPython.display as ipd
from IPython.core.display import display

from PIL import Image
from torchvision.transforms import PILToTensor, ToTensor

from typing import Union

#matplotlib.use('Agg')  # switch backend to run on server

################################################################################
# Plotting utilities for logging and figures
################################################################################


def tensor_to_np(x: torch.Tensor):
    return x.clone().detach().cpu().numpy()


def play_audio(x: torch.Tensor, sample_rate: int = 16000):
    display(ipd.Audio(tensor_to_np(x).flatten(), rate=sample_rate))


def plot_filter(amplitudes: torch.Tensor):
    """
    Given a single set of time-varying filter controls, return plot as image
    """

    amplitudes = amplitudes.clone().detach()

    if amplitudes.ndim == 2:
        magnitudes = amplitudes.cpu().numpy().T
    elif amplitudes.ndim == 3:
        magnitudes = amplitudes[0].cpu().numpy().T
    else:
        raise ValueError("Can only plot single filter response")

    # plot filter controls over time as heatmap
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(magnitudes, aspect='auto')
    fig.colorbar(im, ax=ax)
    ax.invert_yaxis()
    ax.set_title('filter amplitudes')
    ax.set_xlabel('frames')
    ax.set_ylabel('frequency bin')
    plt.tight_layout()

    # save plot to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)

    # return plot as image
    return ToTensor()(np.array(img))


def plot_waveform(x: torch.Tensor, scale: Union[int, float] = 1.0):
    """
    Given single audio waveform, return plot as image
    """
    try:
        assert len(x.shape) == 1 or x.shape[0] == 1
    except AssertionError:
        raise ValueError('Audio input must be single waveform')

    # waveform plot
    fig, ax = plt.subplots(figsize=(8,8))
    fig.subplots_adjust(bottom=0.2)
    plt.xticks(
        #rotation=90
    )
    ax.plot(tensor_to_np(x).flatten(), color='k')
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Waveform Amplitude")
    plt.axis((None, None, -scale, scale))  # set y-axis range

    # save plot to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)

    # return plot as image
    return ToTensor()(np.array(img))


def plot_filter_codebook(x: torch.Tensor, use: torch.Tensor = None):
    """
    Plot a codebook of learned frequency-domain filter controls.
    """

    # scale use rates to [0, 1] for background coloring but not text display
    if use is not None:
        use = use.clone().detach()
        use_normalized = use.clone()
        use_normalized -= use_normalized.min(0, keepdim=True)[0]
        use_normalized /= use_normalized.max(0, keepdim=True)[0]

    n_filters, n_bands = x.shape[0], x.shape[-1]

    # create a square grid layout, which may be partially filled
    grid_size = math.ceil(math.sqrt(n_filters))

    fig, axs = plt.subplots(ncols=grid_size, nrows=grid_size, figsize=(8, 8))

    for i in range(n_filters):
        axis = axs[i//grid_size, i % grid_size]

        # color filter plot according to use rate of filter
        if use is not None:
            assert len(use) == n_filters  # one usage rate per filter
            axis.set_facecolor((1.0, 0.47, 0.42, use_normalized[i].item()))

            x_text = n_bands // 2
            y_text = x[i].max().item() / 2
            axis.text(x_text, y_text, f"{use[i].item() :0.3f}", ha="center", va="center", zorder=10)

        axis.plot(np.zeros(n_bands), 'k', alpha=0.5)  # plot "neutral" line
        axis.plot(tensor_to_np(x[i]).flatten())
        axis.set_xlabel("Frequency")
        axis.set_ylabel("Amplitude")

    plt.tight_layout()

    # save plot to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)

    # return plot as image
    return ToTensor()(np.array(img))


def plot_spectrogram(x: torch.Tensor):
    """
    Given single audio waveform, return spectrogram plot as image
    """
    try:
        assert len(x.shape) == 1 or x.shape[0] == 1
    except AssertionError:
        raise ValueError('Audio input must be single waveform')

    x = x.clone().detach()

    # spectrogram plot
    spec = torch.stft(x.reshape(1, -1),
                      n_fft=512,
                      win_length=512,
                      hop_length=256,
                      window=torch.hann_window(
                          window_length=512
                      ).to(x.device),
                      return_complex=True,
                      center=False
                      )
    spec = torch.squeeze(
        torch.abs(spec) / (torch.max(torch.abs(spec)))
    )  # normalize spectrogram by maximum absolute value

    # save plot to buffer
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pcolormesh(tensor_to_np(torch.log(spec + 1)), vmin=0, vmax=.31)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)

    # return plot image as tensor
    return ToTensor()(np.array(img))


def plot_logits(class_scores: torch.Tensor, target: int = None):
    """
    Given a vector of class scores, and optionally a target index, create a
    simple bar plot of the scores and return as an image
    """

    # require single vector of class scores
    try:
        assert class_scores.ndim <= 1 or class_scores.shape[0] == 1
    except AssertionError:
        raise ValueError('Must provide single vector of class scores')

    # convert to NumPy
    scores = tensor_to_np(class_scores).flatten()
    labels = np.arange(scores.shape[-1])

    # bar plot
    fig = plt.figure(figsize=(8, 8))
    bars = plt.bar(labels, scores, color='k')

    # if target label index is given, highlight corresponding bar
    if target is not None:

        try:
            assert 0 <= target < len(scores)
        except AssertionError:
            raise ValueError("Target must be valid index")

        bars[target].set_color('r')

    # save plot to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)

    # return plot image as tensor
    return ToTensor()(np.array(img))


def get_duration(st: datetime, ed: datetime):
    """Return duration as string"""

    total_seconds = int((ed - st).seconds)
    hours = total_seconds // 3600

    if hours:
        minutes = total_seconds % (3600 * hours) // 60
    else:
        minutes = total_seconds // 60

    seconds = total_seconds
    if minutes:
        seconds = seconds % (60 * minutes)
    if hours:
        seconds = seconds % (3600 * hours)

    duration = ""
    if hours > 0:
        duration += f"{hours}h {minutes}m "
    elif minutes > 0:
        duration += f"{minutes}m "
    duration += f"{seconds}s"

    return duration



