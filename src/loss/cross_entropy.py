import torch
import torch.nn as nn

from src.loss.adversarial import AdversarialLoss

################################################################################
# Cross-entropy loss
################################################################################


class CELoss(AdversarialLoss):
    """
    Measure cross-entropy between categorical (class) distributions
    """
    def __init__(self,
                 targeted: bool = True,
                 reduction: str = 'none',
                 ):
        super().__init__(targeted, reduction)

        self.loss = nn.CrossEntropyLoss(reduction='none')

    def _compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor):

        assert y_pred.device == y_true.device

        assert y_pred.ndim >= 2 and y_pred.shape[-1] >= 2

        if y_true.ndim >= 2:
            y_true = y_true.argmax(dim=-1)

        loss = self.loss(y_pred, y_true)

        if not self.targeted:
            loss *= -1
        return loss
