import torch
import torch.nn.functional as F

from src.loss.adversarial import AdversarialLoss

################################################################################
# Carlini-Wagner loss; measures margin on logits (class scores)
################################################################################


class CWLoss(AdversarialLoss):
    """
    Penalize margin by which undesired class score(s) exceed desired class
    score(s), with a "confidence" parameter determining the margin required to
    incur loss
    """
    def __init__(self,
                 targeted: bool = True,
                 reduction: str = 'none',
                 confidence: float = 0.0,
                 ):
        super().__init__(targeted, reduction)
        self.confidence = confidence

    @staticmethod
    def _one_hot_encode(y: torch.tensor, n_classes: int):
        return F.one_hot(y.type(torch.long), num_classes=n_classes)

    def _compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor):

        assert y_pred.device == y_true.device

        if y_true.shape[-1] != y_pred.shape[-1]:
            y_true = self._one_hot_encode(y_true, y_pred.shape[-1])

        z_target = torch.sum(y_pred * y_true, dim=1)

        z_other = torch.max(
            y_pred * (1 - y_true) + (
                    torch.min(y_pred, dim=-1)[0] - 1
            ).reshape((-1, 1)) * y_true, dim=1,
        )[0]

        if self.targeted:
            loss = torch.clamp(z_other - z_target + self.confidence, min=0.)
        else:
            loss = torch.clamp(z_target - z_other + self.confidence, min=0.)

        return loss
