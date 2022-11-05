import torch
import torch.nn as nn


################################################################################
# Base class for all Loss objects
################################################################################


class Loss(nn.Module):
    """
    Base class for all losses (e.g. classification, auxiliary). Subclasses
    must override the method `_compute_loss()` to compute an unreduced batch
    loss, as batch reduction is left to `forward()`
    """

    def __init__(self,
                 reduction: str = 'none'
                 ):
        super().__init__()

        self.reduction = reduction

    def _compute_loss(self, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, *args, **kwargs):

        batch_loss = self._compute_loss(*args, **kwargs)

        if self.reduction == 'mean':
            return torch.mean(batch_loss)
        elif self.reduction == 'sum':
            return torch.sum(batch_loss)
        elif self.reduction == 'none' or self.reduction is None:
            return batch_loss
        else:
            raise ValueError(f'Invalid reduction {self.reduction}')


