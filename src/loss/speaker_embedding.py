import torch

from src.loss.adversarial import AdversarialLoss
from src.models.speaker.speaker import EmbeddingDistance

################################################################################
# Speaker embedding loss; measures distance in embedding space
################################################################################


class SpeakerEmbeddingLoss(AdversarialLoss):
    def __init__(self,
                 targeted: bool = True,
                 reduction: str = 'none',
                 confidence: float = 0.0,
                 distance_fn: str = 'cosine',
                 threshold: float = 0.0,
                 n_segments: int = 1
                 ):
        super().__init__(targeted, reduction)

        self.confidence = torch.tensor(confidence)
        self.distance_fn = EmbeddingDistance(distance_fn)
        self.threshold = threshold
        self.n_segments = max(n_segments, 1)

    def _compute_loss(
            self,
            y_pred: torch.Tensor,
            y_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Given a batch of predicted and ground truth embeddings, compute
        distance. It is assumed that `n_segments` embeddings have been produced
        from each input audio file, and the distance is taken as the mean over
        all predicted/ground-truth pairs in each tranche of `n_segments`
        embeddings.

        :param y_pred: shape (n_batch, n_segments, embedding_dim)
        :param y_true: shape (n_batch, n_segments, embedding_dim)
        :return: loss, shape (n_batch,)
        """

        dist = self.distance_fn(y_pred, y_true)

        if self.targeted:
            loss = torch.clamp(dist - self.threshold + self.confidence, min=0.)
        else:
            loss = torch.clamp(self.threshold - dist + self.confidence, min=0.)

        return loss


