import torch

from src.attacks.offline.offline import OfflineAttack
from src.pipelines.pipeline import Pipeline
from src.loss.adversarial import AdversarialLoss
from src.utils.writer import Writer

################################################################################
# "Null" attack (apply no perturbations to inputs)
################################################################################


class NullAttack(OfflineAttack):
    """
    Simple baseline attack in which inputs are passed to model unaltered.
    """

    def __init__(self,
                 pipeline: Pipeline,
                 adv_loss: AdversarialLoss,
                 batch_size: int = 1,
                 rand_evals: int = 0,
                 writer: Writer = None,
                 **kwargs
                 ):

        super().__init__(
            pipeline=pipeline,
            adv_loss=adv_loss,
            batch_size=batch_size,
            rand_evals=rand_evals,
            writer=writer
        )

    def _evaluate_batch(self,
                        x: torch.Tensor,
                        y: torch.Tensor,
                        **kwargs
                        ):
        """
        Apply no perturbations to inputs.
        """

        # compute adversarial inputs
        x_adv = x.clone().detach()

        # log attack results
        self._log_step(
            x=x,
            x_adv=x_adv,
            y=y
        )

        return x_adv
