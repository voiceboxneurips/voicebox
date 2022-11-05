import torch
from torch import nn

################################################################################
# Wrapper for all PyTorch audio classifiers
################################################################################


class Model(nn.Module):
    """
    Wrapper class for PyTorch models; provides a consistent interface for
    attack algorithms and prediction
    """

    def __init__(self):
        """
        Initialize model
        """
        super().__init__()

    def forward(self, x: torch.Tensor):
        """
        Perform forward pass
        """
        raise NotImplementedError()

    def load_weights(self, path: str):
        """
        Load weights from checkpoint file
        """
        raise NotImplementedError()

    @staticmethod
    def match_predict(y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Determine whether target pairs are equivalent
        """
        raise NotImplementedError()
