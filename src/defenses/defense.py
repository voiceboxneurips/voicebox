import torch
import torch.nn as nn

from typing import Iterable

from src.defenses.purification import Purification
from src.defenses.detection import Detection
from src.models import Model

################################################################################
# Hold and apply both purification and detection defenses
################################################################################


class Defense(nn.Module):
    """
    Wrapper for sequential application of purification defenses and parallel
    application of detection defenses. Allows for straight-through gradient
    estimation.
    """
    def __init__(self,
                 purification: Iterable[Purification],
                 detection: Iterable[Detection]):
        super().__init__()

        if purification is None:
            self.purification = nn.ModuleList([nn.Identity()])
        else:
            self.purification = nn.ModuleList(purification)

        if detection is None:
            self.detection = None
        else:
            self.detection = nn.ModuleList(detection)

    def purify(self, x: torch.Tensor):
        """
        Apply purification defenses in sequence
        """
        for p in self.purification:

            x = p(x)

        return x

    def detect(self,
               x: torch.Tensor,
               model: Model = None):
        """
        Apply detection defenses in parallel. For each input, return maximum
        score and detection flag obtained from all defenses.
        """

        # require batch dimension
        assert x.ndim >= 2
        n_batch = x.shape[0]

        if isinstance(
                self.detection,
                nn.ModuleList
        ) and len(self.detection) > 0:

            flags = []
            scores = []

            for d in self.detection:

                flag, score = d(x, model)

                assert flag.shape[0] == score.shape[0] == n_batch
                assert torch.prod(flag.shape).item() == n_batch
                assert torch.prod(score.shape).item() == n_batch

                # ensure detector output shape of (n_batch, 1)
                flags.append(flag.reshape(-1, 1))
                scores.append(score.reshape(-1, 1))

            # concatenate outputs, size (n_batch, n_detectors)
            scores = torch.cat(scores, dim=-1)
            flags = torch.cat(flags, dim=-1)

            # final maximum scores/flags, size (n_batch, 1)
            scores = torch.max(scores, dim=-1)[0]
            flags = torch.max(flags, dim=-1)[0]

        else:

            n_batch = x.shape[0]

            # allow zero-gradients to propagate
            flags = x.reshape(n_batch, -1).sum(dim=-1).reshape(n_batch, 1) * 0
            scores = x.reshape(n_batch, -1).sum(dim=-1).reshape(n_batch, 1) * 0

        return flags, scores
