import torch
import torch.nn.functional as F

from src.loss.auxiliary import AuxiliaryLoss

################################################################################
# Control signal losses, for regularizing time-varying controls
################################################################################


class ControlSignalLoss(AuxiliaryLoss):
    """
    Compute losses to regularize time-varying control signals.
    """
    def __init__(self,
                 reduction: str = 'none',
                 loss: str = 'group-sparse-slowness',
                 transpose: bool = False
                 ):

        super().__init__(reduction)

        # select loss variant
        assert loss in ['l2-slowness',
                        'l1-slowness',
                        'group-sparse-slowness',
                        'l1/2-group-sparsity',
                        'l2',
                        'l1'
                        ]
        self.loss = loss
        self.transpose = transpose

    def _compute_loss(self, x: torch.Tensor, x_ref: torch.Tensor = None):
        """Compute specified loss on given control signal"""

        # require (n_batch, time, channels) representation
        assert x.ndim == 3
        b, t, c = x.shape

        # if specified, flip time and channel dimensions
        if self.transpose:
            x = x.permute(0, 2, 1)

        if self.loss == 'l2-slowness':
            loss = (1/((t - 1)*c))*torch.sum(
                        torch.sum(
                            torch.square(
                                torch.diff(x, dim=1)
                            ),
                            dim=2,
                            keepdim=True) + 1e-8,
                    dim=1,
                    keepdim=True
                ).reshape(b)

        elif self.loss == 'l1-slowness':
            loss = (1/((t - 1)*c))*torch.sum(
                torch.sum(
                    torch.abs(
                        torch.diff(x, dim=1)
                    ),
                    dim=2,
                    keepdim=True) + 1e-8,
                dim=1,
                keepdim=True
            ).reshape(b)

        elif self.loss == 'group-sparse-slowness':
            loss = (1/((t - 1)*c))*torch.square(
                torch.sum(
                    torch.sqrt(
                        torch.sum(
                            torch.square(
                                torch.diff(x, dim=1)
                            ),
                            dim=2,
                            keepdim=True) + 1e-8
                    ),
                    dim=1,
                    keepdim=True
                )
            ).reshape(b)

        elif self.loss == 'l1/2-group-sparsity':
            loss = (1/((t - 1)*c))*torch.sum(
                torch.sum(
                    torch.abs(
                        torch.diff(x, dim=1) + 1e-8
                    )**0.5,
                    dim=2,
                    keepdim=True
                )**2,
                dim=1,
                keepdim=True
            ).reshape(b)

        elif self.loss == 'l2':
            loss = x.norm(dim=(1, 2), p=2).reshape(b)

        elif self.loss == 'l1':
            loss = x.norm(dim=(1, 2), p=1).reshape(b)

        else:
            raise ValueError(f'Invalid control-signal loss {self.loss}')

        return loss

    def set_reference(self, x_ref: torch.Tensor):
        pass
