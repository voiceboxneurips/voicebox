import torch
import torch.nn.functional as F

import math

################################################################################
# Resampling utilities for DEMUCS architecture
################################################################################


def sinc(x: torch.Tensor):
    """
    Sinc function.
    """
    return torch.where(
        x == 0,
        torch.tensor(1., device=x.device, dtype=x.dtype),
        torch.sin(x) / x
    )


def kernel_upsample2(zeros=56):
    """
    Compute windowed sinc kernel for upsampling by a factor of 2.
    """
    win = torch.hann_window(4 * zeros + 1, periodic=False)
    win_odd = win[1::2]
    t = torch.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros)
    t *= math.pi
    kernel = (sinc(t) * win_odd).view(1, 1, -1)
    return kernel


def upsample2(x, zeros=56):
    """
    Upsample input by a factor of 2 using sinc interpolation.
    """
    *other, time = x.shape
    kernel = kernel_upsample2(zeros).to(x)
    out = F.conv1d(
        x.view(-1, 1, time),
        kernel,
        padding=zeros
    )[..., 1:].view(*other, time)
    y = torch.stack([x, out], dim=-1)
    return y.view(*other, -1)


def kernel_downsample2(zeros=56):
    """
    Compute windowed sinc kernel for downsampling by a factor of 2.
    """
    win = torch.hann_window(4 * zeros + 1, periodic=False)
    win_odd = win[1::2]
    t = torch.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros)
    t.mul_(math.pi)
    kernel = (sinc(t) * win_odd).view(1, 1, -1)
    return kernel


def downsample2(x, zeros=56):
    """
    Downsample input by a factor of 2 using sinc interpolation.
    """
    if x.shape[-1] % 2 != 0:
        x = F.pad(x, (0, 1))
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    *other, time = x_odd.shape
    kernel = kernel_downsample2(zeros).to(x)
    out = x_even + F.conv1d(
        x_odd.view(-1, 1, time),
        kernel,
        padding=zeros
    )[..., :-1].view(*other, time)
    return out.view(*other, -1).mul(0.5)
