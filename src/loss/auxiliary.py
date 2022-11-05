import torch

from src.loss.loss import Loss

################################################################################
# Base class for adversarial auxiliary objectives
################################################################################


class AuxiliaryLoss(Loss):
    """
    Wrapper for auxiliary (e.g. perceptual) losses, computed either on inputs
    alone ("reference-free") or input-reference pairs ("full reference").
    Subclasses must override the method `_compute_loss()` to compute an
    unreduced batch loss, as batch reduction is left to `forward()`.

    Subclasses must also implement the method `set_reference()`, which can be
    used to compute and cache references. This may be useful in avoiding
    re-computing expensive reference representations, such as the psychoacoustic
    thresholds required by a frequency-masking loss.
    """
    def __init__(self,
                 reduction: str = 'none'
                 ):
        super().__init__(reduction)

    def _compute_loss(self, x: torch.Tensor, x_ref: torch.Tensor = None):
        """
        Compute unreduced batch loss.

        :param x: input, shape (n_batch, ...)
        :param x_ref: reference, shape (n_batch, ...)
        :return: loss, shape (n_batch,)
        """
        raise NotImplementedError()

    @staticmethod
    def _check_broadcastable(x: torch.Tensor, x_ref: torch.Tensor):
        """
        Check whether input and reference tensors are broadcastable
        """

        broadcastable = all(
            (m == n) or (m == 1) or (n == 1) for m, n in zip(
                x.shape[::-1], x_ref.shape[::-1]
            )
        )

        # broadcast cannot expand input batch dimension
        valid = x.shape[0] == x_ref.shape[0] or x_ref.shape[0] == 1

        return broadcastable * valid

    def set_reference(self, x_ref: torch.Tensor):
        """
        Compute and cache reference representation(s).
        """
        raise NotImplementedError()

