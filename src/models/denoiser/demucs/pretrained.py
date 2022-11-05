import torch
from pathlib import Path

from src.constants import MODELS_DIR
from src.models.denoiser.demucs.demucs import Demucs

################################################################################
# Load pretrained DEMUCS U-Net denoiser
################################################################################


def load_demucs(name: str, pretrained: bool = True):
    """Load a pretrained Demucs denoiser model by name."""

    # causal, hidden dimension 48, trained on DNS dataset
    if name == 'dns_48':
        config = {
            'hidden_dim': 48,
            'depth': 5,
            'resample': 4,
            'stride_conv': 4,
            'kernel_conv': 8,
            'growth': 1.0,
            'causal': True,
            'normalize': True
        }
        path = Path(MODELS_DIR) / 'denoiser' / 'demucs' / 'dns_48.pt'

    # causal, hidden dimension 64, trained on DNS dataset
    elif name == 'dns_64':
        config = {
            'hidden_dim': 64,
            'depth': 5,
            'resample': 4,
            'stride_conv': 4,
            'kernel_conv': 8,
            'growth': 1.0,
            'causal': True,
            'normalize': True
        }
        path = Path(MODELS_DIR) / 'denoiser' / 'demucs' / 'dns_64.pt'
        raise NotImplementedError(f'Demucs model `dns_64` not currently '
                                  f'supported due to file size')

    # causal, hidden dimension 64
    elif name == 'master_64':
        config = {
            'hidden_dim': 64,
            'depth': 5,
            'resample': 4,
            'stride_conv': 4,
            'kernel_conv': 8,
            'growth': 1.0,
            'causal': True,
            'normalize': True
        }
        path = Path(MODELS_DIR) / 'denoiser' / 'demucs' / 'master_64.pt'
        raise NotImplementedError(f'Demucs model `master_64` not currently '
                                  f'supported due to file size')

    # non-causal, hidden dimension 64, trained on Valentini dataset
    elif name == 'valentini_nc':
        config = {
            'hidden_dim': 64,
            'depth': 5,
            'resample': 2,
            'stride_conv': 2,
            'kernel_conv': 8,
            'growth': 1.0,
            'causal': False,
            'normalize': True
        }
        path = Path(MODELS_DIR) / 'denoiser' / 'demucs' / 'valentini_nc.pt'
        raise NotImplementedError(f'Demucs model `valentini_nc` not currently '
                                  f'supported due to file size')

    elif name == 'experimental_small':

        config = {
            'hidden_dim': 48,
            'depth': 3,
            'resample': 4,
            'stride_conv': 4,
            'kernel_conv': 8,
            'growth': 1.0,
            'causal': True,
            'normalize': True,
            'original': False,
            'use_bias': False
        }
        path = None

    else:
        raise ValueError(f'Invalid model name {name}')

    # initialize model
    model = Demucs(**config)

    # load pretrained weights from checkpoint file
    if pretrained and path is not None:
        model.load_state_dict(torch.load(path))

    return model
