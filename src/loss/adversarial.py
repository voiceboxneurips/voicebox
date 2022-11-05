import torch

from src.loss.loss import Loss

################################################################################
# Base class for adversarial classification objectives
################################################################################


class AdversarialLoss(Loss):
    """
    Wrapper for adversarial losses computed on paired targets. Subclasses must
    override the method `_compute_loss()` to compute an unreduced batch loss, as
    batch reduction is left to `forward()`
    """

    def __init__(self,
                 targeted: bool = True,
                 reduction: str = 'none'
                 ):
        super().__init__(reduction)
        self.targeted = targeted

    def _compute_loss(self, *args, **kwargs):
        raise NotImplementedError()

