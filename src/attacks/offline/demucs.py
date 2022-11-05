import torch

from src.models.denoiser import load_demucs
from src.attacks.offline.trainable import TrainableAttack
from src.pipelines.pipeline import Pipeline
from src.loss.adversarial import AdversarialLoss

################################################################################
# Attack using Demucs waveform-convolutional U-net
################################################################################


class DemucsAttack(TrainableAttack):
    """
    Train a Demucs model to apply adversarial transformations to incoming audio
    """
    def __init__(self,
                 pipeline: Pipeline,
                 adv_loss: AdversarialLoss,
                 model_name: str = 'dns_48',
                 pretrained: bool = True,
                 **kwargs):
        """
        Train a pre-trained Demucs model to apply adversarial transformations to
        incoming audio. Gradient descent variants adapted from `Evading
        Adversarial Example Detection Defenses with Orthogonal Projected
        Gradient Descent` by Bryniarski et al.
        (https://github.com/v-wangg/OrthogonalPGD).

        :param pipeline: a Pipeline object wrapping a (defended) classifier
        :param adv_loss: a Loss object encapsulating the adversarial objective;
                         should take model predictions and targets as arguments
        :param aux_loss: an auxiliary Loss object; should take original and
                         adversarial inputs as arguments
        :param model_name: name of pre-trained Demucs model to load
        :param pretrained: if True, seek pretrained weights for given model
        :param opt: optimizer; must be one of 'adam', 'sgd', or 'lbfgs'
        :param lr: perturbation learning rate
        :param mode: PGD variant; must be one of None, 'orthogonal', 'selective'
        :param project_grad: p-norm for gradient regularization; must be one of
                             inf, 2, or None
        :param k: if not None, perform gradient projection every kth step
        :param max_iter: the maximum number of iterations per batch
        :param epochs: optimization epochs over training data
        :param eot_iter: resampling interval for Pipeline simulation parameters;
                         if 0 or None, do not resample parameters
        :param batch_size: batch size for attack
        :param rand_evals: randomly-resampled simulated evaluations per each
                           final generated attack
        :param writer: a Writer object for logging attack progress
        """
        super().__init__(
            pipeline=pipeline,
            adv_loss=adv_loss,
            perturbation=load_demucs(model_name, pretrained),
            **kwargs
        )
