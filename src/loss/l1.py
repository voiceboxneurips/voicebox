import torch

from src.loss.auxiliary import AuxiliaryLoss

################################################################################
# L1 loss
################################################################################


class L1Loss(AuxiliaryLoss):

    def __init__(self,
                 reduction: str = 'none',
                 ):
        super().__init__(reduction)
        self.ref_wav = None

    def set_reference(self, x_ref: torch.Tensor):
        self.ref_wav = x_ref.clone().detach()

    def _compute_loss(self, x: torch.Tensor, x_ref: torch.Tensor = None):
        """
        Compute L1 distance between input and reference. If no reference
        is provided, a stored reference will be used. If no stored reference is
        available, the L1 norm of the input will be returned.

        :param x: input tensor of shape (n_batch, ...)
        :param x_ref: reference tensor of shape (n_batch, ...) or (1, ...)
        :return:
        """

        # if no reference is stored or provided, apply L2 norm to input directly
        if x_ref is None and self.ref_wav is None:
            x_ref = torch.zeros_like(x)

        # use stored reference if none provided
        elif x_ref is None:
            x_ref = self.ref_wav

        # ensure broadcastable inputs
        assert self._check_broadcastable(
            x, x_ref
        ), f"Cannot broadcast inputs of shape {x.shape} " \
           f"with reference of shape {x_ref.shape}"

        assert x.ndim >= 2  # require batch dimension
        n_batch = x.shape[0]

        return torch.mean(
            (x - x_ref).abs().reshape(n_batch, -1),
            dim=-1
        )

