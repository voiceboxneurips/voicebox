import torch
from torch.utils.data import Dataset

import warnings

from typing import Union, List, Tuple

from src.attacks.offline.trainable import TrainableAttack
from src.attacks.offline.perturbation.perturbation import Perturbation
from src.attacks.offline.perturbation.white_noise import WhiteNoisePerturbation
from src.pipelines.pipeline import Pipeline
from src.loss.adversarial import AdversarialLoss

################################################################################
# White noise attack
################################################################################


class WhiteNoiseAttack(TrainableAttack):
    """
    Simple baseline attack in which white noise is added to inputs.
    """
    def __init__(self,
                 pipeline: Pipeline,
                 adv_loss: AdversarialLoss,
                 snr_low: float = -10.0,
                 snr_high: float = 60.0,
                 step_size: float = 5.0,
                 min_success_rate: float = 0.9,
                 search: str = 'bisection',
                 **kwargs
                 ):
        """
        Sweep a range of SNR values to find best signal-to-noise (SNR) ratio
        at which to apply noise to inputs; adapted from https://bit.ly/3tcDF7u.

        :param pipeline: Pipeline object
        :param adv_loss: AdversarialLoss object
        :param snr: signal-to-noise ratio of attack. Can be a float, in which
                    case noise will be applied at the given SNR, or a pair of
                    floats representing the endpoints of the search space
        :param step: step size for search over SNR values
        :param search: search method if range of SNR values is given; must be
                       one of 'linear', 'bisection', or 'none'
        """
        self.snr_low = snr_low
        self.snr_high = snr_high
        self.step_size = step_size
        self.search = search
        self.min_success_rate = min_success_rate

        super().__init__(
            pipeline=pipeline,
            adv_loss=adv_loss,
            perturbation=WhiteNoisePerturbation(snr=snr_low),
            **kwargs
        )

    @torch.no_grad()
    def train(self,
              x_train: torch.Tensor = None,
              y_train: torch.Tensor = None,
              data_train: Dataset = None,
              x_val: torch.Tensor = None,
              y_val: torch.Tensor = None,
              data_val: Dataset = None,
              *args,
              **kwargs
              ):
        """
        Perform a single epoch of "training" by sweeping for an optimal SNR
        value over the training data.
        """

        loader_train, loader_val = self._prepare_data(
            x_train,
            y_train,
            data_train,
            x_val,
            y_val,
            data_val)

        # match devices and set reference if necessary
        ref_batch = next(iter(loader_train))

        if isinstance(ref_batch, tuple):
            x_ref = ref_batch[0]
            warnings.warn('Warning: provided dataset yields batches in tuple '
                          'format; the first two tensors of each batch will be '
                          'interpreted as inputs and targets, respectively, '
                          'and any remaining tensors will be ignored. To pass '
                          'additional named tensor arguments, use a dictionary '
                          'batch format with keys `x` and `y` for inputs and '
                          'targets, respectively.')
        elif isinstance(ref_batch, dict):
            x_ref = ref_batch['x']
        else:
            x_ref = ref_batch

        if hasattr(self.perturbation, "set_reference"):
            try:
                self.perturbation.set_reference(
                    x_ref.to(self.pipeline.device))
            except AttributeError:
                pass

        # enumerate possible SNR values for search
        snr_values = torch.arange(self.snr_low, self.snr_high, self.step_size)

        # track iterations
        self._iter_id = 0
        self._batch_id = 0
        self._epoch_id = 0

        # avoid unnecessary search
        if self.snr_low == self.snr_high \
                or len(snr_values) < 2 \
                or self.search in ['none', None]:
            self.perturbation.set_snr(self.snr_low)

        else:

            # find best SNR via search
            i_min = 0
            i_max = len(snr_values)
            snr_best = self.snr_low

            # perform bisection search for maximum SNR value which achieves
            # minimum success threshold
            if self.search == 'bisection':

                while i_min < i_max:

                    # determine midpoint index
                    i_mid = (i_min + i_max) // 2
                    snr = snr_values[i_mid]

                    # set SNR
                    self.perturbation.set_snr(snr)

                    # compute success rate over training data at each candidate
                    # SNR level
                    successes = 0
                    n = 0

                    self._batch_id = 0
                    for batch in loader_train:

                        if isinstance(batch, dict):
                            x, y = batch['x'], batch['y']
                        else:
                            x, y, *_ = batch

                        x = x.to(self.pipeline.device)
                        y = y.to(self.pipeline.device)

                        n += len(x)
                        x_adv = self.perturbation(x)
                        outputs = self.pipeline(x_adv)
                        adv_scores = self.adv_loss(outputs, y)
                        adv_loss = adv_scores.mean()

                        batch_successes = (1.0 * self._compute_success_array(
                            x, y, x_adv)).sum().item()
                        successes += batch_successes

                        self._log_step(
                            x,
                            x_adv,
                            y,
                            adv_loss,
                            success_rate=batch_successes/len(x)
                        )

                        self._batch_id += 1
                        self._iter_id += 1

                    success_rate = successes / n

                    if success_rate >= self.min_success_rate:
                        snr_best = snr
                        i_min = i_mid + 1
                    else:
                        i_max = i_mid

            # perform linear search for SNR level
            elif self.search == 'linear':

                for snr in snr_values:

                    # set SNR
                    self.perturbation.set_snr(snr)

                    # compute success rate over training data at each candidate
                    # SNR level
                    successes = 0
                    n = 0

                    self._batch_id = 0
                    for batch in loader_train:

                        if isinstance(batch, dict):
                            x, y = batch['x'], batch['y']
                        else:
                            x, y, *_ = batch

                        x = x.to(self.pipeline.device)
                        y = y.to(self.pipeline.device)

                        n += len(x)
                        x_adv = self.perturbation(x)
                        outputs = self.pipeline(x_adv)
                        adv_scores = self.adv_loss(outputs, y)
                        adv_loss = adv_scores.mean()
                        batch_successes = (1.0 * self._compute_success_array(
                            x, y, x_adv)).sum().item()
                        successes += batch_successes

                        self._log_step(
                            x,
                            x_adv,
                            y,
                            adv_loss,
                            success_rate=batch_successes/len(x)
                        )

                        self._batch_id += 1
                        self._iter_id += 1

                    success_rate = successes / n

                    if success_rate >= self.min_success_rate:
                        snr_best = snr
            else:
                raise ValueError(f'Invalid search method {self.search}')

            # set final SNR value
            self.perturbation.set_snr(snr_best)

        # perform validation
        adv_scores = []
        aux_scores = []
        det_scores = []
        success_indicators = []
        detection_indicators = []

        self.perturbation.eval()

        for batch_id, batch in enumerate(loader_val):

            # randomize simulation for each validation batch
            self.pipeline.sample_params()

            if isinstance(batch, dict):
                x_orig, targets = batch['x'], batch['y']
            else:
                x_orig, targets, *_ = batch

            n_batch = x_orig.shape[0]

            x_orig = x_orig.to(self.pipeline.device)
            targets = targets.to(self.pipeline.device)

            # set reference for auxiliary loss
            self._set_loss_reference(x_orig)

            with torch.no_grad():

                # compute adversarial loss
                x_adv = self._evaluate_batch(x_orig, targets)
                outputs = self.pipeline(x_adv)
                adv_scores.append(self.adv_loss(outputs, targets).flatten())

                # compute adversarial success rate
                success_indicators.append(
                    1.0 * self._compute_success_array(
                        x_orig, targets, x_adv
                    ).flatten())

                # compute defense loss and detection indicators
                def_results = self.pipeline.detect(x_adv)
                detection_indicators.append(1.0 * def_results[0].flatten())
                det_scores.append(def_results[1].flatten())

                # compute auxiliary loss
                if self.aux_loss is not None:
                    aux_scores.append(
                        self._compute_aux_loss(x_adv).flatten())
                else:
                    aux_scores.append(torch.zeros(n_batch))

        tag = f'{self.__class__.__name__}-' \
              f'{self.aux_loss.__class__.__name__}'

        if self.writer is not None:

            with self.writer.force_logging():

                # adversarial loss value
                self.writer.log_scalar(
                    torch.cat(adv_scores, dim=0).mean(),
                    f"{tag}/adversarial-loss-val",
                    global_step=self._iter_id
                )

                # detector loss value
                self.writer.log_scalar(
                    torch.cat(det_scores, dim=0).mean(),
                    f"{tag}/detector-loss-val",
                    global_step=self._iter_id
                )

                # auxiliary loss value
                self.writer.log_scalar(
                    torch.cat(aux_scores, dim=0).mean(),
                    f"{tag}/auxiliary-loss-val",
                    global_step=self._iter_id
                )

                # adversarial success rate
                self.writer.log_scalar(
                    torch.cat(success_indicators, dim=0).mean(),
                    f"{tag}/success-rate-val",
                    global_step=self._iter_id
                )

                # adversarial detection rate
                self.writer.log_scalar(
                    torch.cat(detection_indicators, dim=0).mean(),
                    f"{tag}/detection-rate-val",
                    global_step=self._iter_id
                )

        # freeze model parameters
        self.perturbation.eval()
        for p in self.perturbation.parameters():
            p.requires_grad = False

        # save model/perturbation
        self._checkpoint()

    def _evaluate_batch(self,
                        x: torch.Tensor,
                        y: torch.Tensor,
                        **kwargs
                        ):
        """
        Apply white noise perturbations to inputs.
        """

        # require batch dimension
        assert x.ndim >= 2

        return self.perturbation(x)
