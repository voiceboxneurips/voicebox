import os
import warnings

import torch
import torch.nn as nn

from pathlib import Path
from typing import Tuple, Union

from torch.utils.data import Dataset, DataLoader

from src.attacks.offline.offline import OfflineAttack
from src.attacks.offline.orthogonal_selective import SelectiveOrthogonalPGDMixin
from src.attacks.offline.perturbation.perturbation import Perturbation
from src.pipelines import Pipeline
from src.loss.adversarial import AdversarialLoss
from src.loss.auxiliary import AuxiliaryLoss
from src.utils.writer import Writer

################################################################################
# Base class for trainable attacks
################################################################################


class TrainableAttack(OfflineAttack, SelectiveOrthogonalPGDMixin):

    def __init__(self,
                 pipeline: Pipeline,
                 perturbation: torch.nn.Module,
                 adv_loss: AdversarialLoss,
                 aux_loss: AuxiliaryLoss = None,
                 adv_success_thresh: float = 0.0,
                 det_success_thresh: float = 0.0,
                 opt: str = 'adam',
                 lr: float = 1e-4,
                 pgd_variant: str = None,
                 pgd_norm: Union[str, int, float] = None,
                 scale_grad: Union[int, float, str] = None,
                 k: int = None,
                 epochs: int = 10,
                 max_iter: int = 1,
                 batch_size: int = 32,
                 rand_evals: int = 0,
                 eot_iter: int = 0,
                 checkpoint_name: str = None,
                 writer: Writer = None,
                 validate: bool = True,
                 **kwargs):

        super().__init__(
            pipeline=pipeline,
            adv_loss=adv_loss,
            aux_loss=aux_loss,
            batch_size=batch_size,
            rand_evals=rand_evals,
            writer=writer,
            **kwargs
        )

        # underlying perturbation/model
        self.perturbation = perturbation.to(self.pipeline.device)

        # optimizer
        self.lr = lr
        self.opt = opt
        self.optimizer = None
        self.epochs = epochs
        self.max_iter = max_iter
        self.eot_iter = eot_iter

        # PGD algorithm
        self.pgd_variant = pgd_variant
        self.pgd_norm = pgd_norm
        self.scale_grad = scale_grad
        self.k = k
        self.adv_success_thresh = adv_success_thresh
        self.det_success_thresh = det_success_thresh

        # determine whether to perform validation during training
        self.validate = validate

        # checkpointing
        self.checkpoint_name = checkpoint_name

        # track epoch count
        self._epoch_id = 0

        self._check_loss()

    def _tile_and_create_dataset(self, x: torch.Tensor, y: torch.Tensor):
        """
        Given inputs and targets, create a dataset. If only a single target is
        given, repeat to match length of inputs.
        """
        # if only a single target is given, repeat to length of dataset
        y = y.unsqueeze(0) if y.ndim < 1 else y

        if y.shape[0] == 1:
            y = y.repeat_interleave(dim=0, repeats=x.shape[0])

        return self._create_dataset(x, y)

    def _get_optimizer(self):
        """Configure optimizer for stored model/perturbation"""

        if self.opt == 'adam':
            optimizer = torch.optim.Adam(
                self.perturbation.parameters(),
                lr=self.lr,
                betas=(.99, .999),
                eps=1e-7,
                amsgrad=False
            )
        elif self.opt == 'lbfgs':
            optimizer = torch.optim.LBFGS(
                self.perturbation.parameters(),
                lr=self.lr,
                line_search_fn='strong_wolfe'
            )
        elif self.opt == 'sgd':
            optimizer = torch.optim.SGD(
                self.perturbation.parameters(),
                lr=self.lr
            )
        else:
            raise ValueError(f'Invalid optimizer {self.opt}')

        return optimizer

    def _set_loss_reference(self, x: torch.Tensor):
        """
        Pass reference audio to auxiliary loss to avoid re-computing expensive
        intermediate representations
        """
        if self.aux_loss is not None:
            self.aux_loss.set_reference(x)

    def _compute_aux_loss(self,
                          x_adv: torch.Tensor,
                          x_ref: torch.Tensor = None):
        """Compute auxiliary loss given perturbed input"""
        return self.aux_loss(x_adv, x_ref)

    def _prepare_data(self,
                      x_train: torch.Tensor = None,
                      y_train: torch.Tensor = None,
                      data_train: Dataset = None,
                      x_val: torch.Tensor = None,
                      y_val: torch.Tensor = None,
                      data_val: Dataset = None,
                      ):

        # require training dataset
        assert (x_train is not None and y_train is not None) \
               or data_train is not None, 'Must provide training data'

        # require validation dataset
        assert (x_val is not None and y_val is not None) \
               or data_val is not None, 'Must provide validation data'

        # package tensors as datasets
        if data_train is None:
            data_train = self._tile_and_create_dataset(x_train, y_train)
        if data_val is None:
            data_val = self._tile_and_create_dataset(x_val, y_val)

        loader_train = DataLoader(
            dataset=data_train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers
        )

        loader_val = DataLoader(
            dataset=data_val,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers
        )

        return loader_train, loader_val

    def _train_batch(self,
                     x: torch.Tensor,
                     y: torch.Tensor,
                     *args,
                     **kwargs):
        """Optimize stored model/perturbation over a batch of inputs"""

        # require batch dimension
        assert x.ndim >= 2
        n_batch = x.shape[0]

        x = x.detach()

        # set reference for auxiliary loss to avoid re-computing
        self._set_loss_reference(x)

        # randomly sample simulation parameters
        if self.eot_iter and not self._iter_id % self.eot_iter:
            self.pipeline.sample_params()

        def closure():

            # placeholder for final model/perturbation gradients
            model_gradients = \
                self._retrieve_parameter_gradients(self.perturbation)
            grad_total = torch.zeros_like(model_gradients)

            # apply adversarial perturbation to batch and obtain predictions
            perturbed = self.perturbation(x, *args, **kwargs)
            outputs = self.pipeline(perturbed)

            # reset parameter gradients, using `None` for performance boost
            self.perturbation.zero_grad(set_to_none=True)

            # compute flattened parameter gradients w.r.t. adversarial loss
            adv_scores = self.adv_loss(outputs, y)
            adv_loss = torch.mean(adv_scores)
            adv_loss.backward(retain_graph=True)
            adv_loss_grad = self._retrieve_parameter_gradients(
                self.perturbation
            ).detach()

            # reset parameter gradients, using `None` for performance boost
            self.perturbation.zero_grad(set_to_none=True)

            # compute flattened parameter gradients w.r.t. detector loss
            detector_flags, detector_scores = self.pipeline.detect(perturbed)
            detector_loss = torch.mean(detector_scores)
            detector_loss.backward(retain_graph=True)
            detector_loss_grad = self._retrieve_parameter_gradients(
                self.perturbation
            ).detach()

            # reset parameter gradients, using `None` for performance boost
            self.perturbation.zero_grad(set_to_none=True)

            # compute flattened parameter gradients w.r.t. auxiliary loss
            if self.aux_loss is not None:
                aux_scores = self._compute_aux_loss(perturbed)
                aux_loss = torch.mean(aux_scores)
                aux_loss.backward()
                aux_loss_grad = self._retrieve_parameter_gradients(
                    self.perturbation
                ).detach()
            else:  # if no auxiliary loss, do not penalize
                aux_scores = torch.zeros(n_batch).to(x.device)
                aux_loss = torch.mean(aux_scores)
                aux_loss_grad = torch.zeros_like(adv_loss_grad).detach()

            # classifier evasion indicator, reshape for broadcasting
            adv_success = (adv_loss <= self.adv_success_thresh) * 1.0

            # detector evasion indicator, reshape for broadcasting
            detector_success = (detector_loss <= self.det_success_thresh) * 1.0

            # perform standard, orthogonal, or selective gradient
            # accumulation
            if self.pgd_variant is None or self.pgd_variant == 'none':

                # for standard PGD, sum loss gradients
                grad_total += adv_loss_grad + \
                              detector_loss_grad + \
                              aux_loss_grad

            elif self.pgd_variant == 'orthogonal':

                # for orthogonal PGD, orthogonalize loss gradients and
                # select one for update; optionally, orthogonalize only
                # every kth step
                if self.k and self._iter_id % self.k:
                    adv_loss_grad_proj = adv_loss_grad
                    detector_loss_grad_proj = detector_loss_grad
                    aux_loss_grad_proj = aux_loss_grad
                else:
                    adv_loss_grad_proj = self._component_orthogonal(
                        adv_loss_grad,
                        detector_loss_grad,
                        aux_loss_grad
                    )
                    detector_loss_grad_proj = self._component_orthogonal(
                        detector_loss_grad,
                        adv_loss_grad,
                        aux_loss_grad
                    )
                    aux_loss_grad_proj = self._component_orthogonal(
                        aux_loss_grad,
                        detector_loss_grad,
                        adv_loss_grad
                    )

                # update 'along' a single loss gradient per iteration
                grad_total += adv_loss_grad_proj * (1 - adv_success)
                grad_total += detector_loss_grad_proj * adv_success \
                              * (1 - detector_success)
                grad_total += aux_loss_grad_proj * adv_success * \
                              detector_success

            elif self.pgd_variant == 'selective':

                # only consider a single loss per iteration, without
                # ensuring orthogonality to remaining loss gradients
                grad_total += adv_loss_grad * (1 - adv_success)
                grad_total += detector_loss_grad * adv_success \
                              * (1 - detector_success)
                grad_total += aux_loss_grad * adv_success * detector_success

            else:
                raise ValueError(f'Invalid attack mode {self.pgd_variant}')

            # regularize gradients via p-norm projection
            if self.scale_grad in [2, float(2), "2"]:
                grad_norms = torch.norm(
                    grad_total, p=2, dim=-1
                ) + 1e-20
                grad_total = grad_total / grad_norms
            elif self.scale_grad in [float("inf"), "inf"]:
                grad_total = torch.sign(grad_total)
            elif self.scale_grad in ['none', None]:
                pass
            else:
                raise ValueError(f'Invalid gradient regularization norm '
                                 f'{self.scale_grad}'
                                 )

            # set final parameter gradients
            self._set_parameter_gradients(
                grad_total.flatten(),
                self.perturbation
            )

            # log results
            if self.writer is not None:
                self._log_step(
                    x=x,
                    x_adv=perturbed,
                    y=y,
                    adv_loss=adv_loss,
                    det_loss=detector_loss,
                    aux_loss=aux_loss,
                    detection_rate=torch.mean(1.0 * detector_flags)
                )

            # return placeholder loss
            return adv_loss + detector_loss + aux_loss

        # optimizer step, using stored gradients
        self.optimizer.step(closure)

        # project perturbation to feasible region
        if hasattr(self.perturbation, "project_valid"):
            try:
                self.perturbation.project_valid()
            except AttributeError:
                pass

        # update total iteration count
        self._iter_id += 1

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
        Optimize trainable attack parameters over training data.

        Parameters
        ----------

        Returns
        -------
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

        # configure optimizer
        self.optimizer = self._get_optimizer()

        # reset cumulative iteration count
        self._iter_id = 0

        # optimize perturbation over given number of epochs
        for epoch_id in range(self.epochs):

            self._batch_id = 0
            self._epoch_id = epoch_id

            self.perturbation.train()
            for batch_id, batch in enumerate(loader_train):

                self._batch_id = batch_id

                # allow for different dataset formats
                if isinstance(batch, tuple):
                    batch = {
                        'x': batch[0],
                        'y': batch[1]
                    }

                # match devices
                for k in batch.keys():
                    batch[k] = batch[k].to(self.pipeline.device)

                self._train_batch(**batch)

            # perform validation once per epoch
            if self.validate:
                adv_scores = []
                aux_scores = []
                det_scores = []
                success_indicators = []
                detection_indicators = []

                self.perturbation.eval()
                for batch_id, batch in enumerate(loader_val):

                    # randomize simulation for each validation batch
                    self.pipeline.sample_params()

                    # allow for different dataset formats
                    if isinstance(batch, tuple):
                        batch = {
                            'x': batch[0],
                            'y': batch[1]
                        }

                    n_batch = batch['x'].shape[0]

                    # match devices
                    for k in batch.keys():
                        batch[k] = batch[k].to(self.pipeline.device)

                    # set reference for auxiliary loss
                    self._set_loss_reference(batch['x'])

                    with torch.no_grad():

                        # compute adversarial loss
                        x_adv = self._evaluate_batch(**batch)
                        outputs = self.pipeline(x_adv)
                        adv_scores.append(self.adv_loss(outputs, batch['y']).flatten())

                        # compute adversarial success rate
                        success_indicators.append(
                            1.0 * self._compute_success_array(
                                x=batch['x'], y=batch['y'], x_adv=x_adv
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

        # clear optimizer
        self.optimizer = None

        # freeze model parameters
        self.perturbation.eval()
        for p in self.perturbation.parameters():
            p.requires_grad = False

        # save model/perturbation
        self._checkpoint()

    def _evaluate_batch(self,
                        x: torch.Tensor,
                        y: torch.Tensor,
                        *args,
                        **kwargs
                        ):
        """Evaluate batch of inputs by passing through model/perturbation"""

        x_orig = x.clone().detach()
        x_adv = self.perturbation(x_orig, *args, **kwargs)
        return x_adv

    @torch.no_grad()
    def evaluate(self,
                 x: torch.Tensor = None,
                 y: torch.Tensor = None,
                 dataset: Dataset = None,
                 *args,
                 **kwargs
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        self.perturbation.eval()
        return super().evaluate(x, y, dataset, *args, **kwargs)

    def _log_step(self,
                  x: torch.Tensor,
                  x_adv: torch.Tensor,
                  y: torch.Tensor,
                  adv_loss: Union[float, torch.Tensor] = None,
                  det_loss: Union[float, torch.Tensor] = None,
                  aux_loss: Union[float, torch.Tensor] = None,
                  success_rate: Union[float, torch.Tensor] = None,
                  detection_rate: Union[float, torch.Tensor] = None,
                  idx: int = 0,
                  tag: str = None,
                  *args,
                  **kwargs
                  ):
        """
        Log attack progress.

        Parameters
        ----------
        x (torch.Tensor):       batch of original inputs
        x_adv (torch.Tensor):   batch of adversarial inputs
        y (torch.Tensor):       batch of targets
        adv_loss (float):       adversarial loss value
        aux_loss (float):       auxiliary loss value
        det_loss (float):       detector loss value
        success_rate (float):   attack success rate
        detection_rate (float): attack detection rate
        idx (int):              batch index for logging individual examples
        tag (str):              label for logging output
        """

        if self.writer is None or self._iter_id % self.writer.log_iter:
            return

        if tag is None:
            tag = f'{self.__class__.__name__}-' \
                  f'{self.aux_loss.__class__.__name__}'

        super()._log_step(
            x,
            x_adv,
            y,
            adv_loss=adv_loss,
            det_loss=det_loss,
            aux_loss=aux_loss,
            success_rate=success_rate,
            detection_rate=detection_rate,
            idx=idx,
            tag=tag
        )

        # log perturbation visualizations
        if hasattr(self.perturbation, "visualize"):
            try:
                visualizations = self.perturbation.visualize()  # Dict[str: tensor]
                for name, image in visualizations.items():
                    self.writer.log_image(
                        tag=f'{tag}/{name}',
                        image=image,
                        global_step=self._iter_id
                    )
            except AttributeError:
                pass

    def load(self, path: Union[str, Path]):
        """Load weights for stored perturbation/model"""

        checkpoint_path = Path(path)

        # for files, load directly
        if checkpoint_path.is_file():
            final_path = checkpoint_path

        # for directory, check for most recent file
        elif checkpoint_path.is_dir():

            # search for files with matching identifier
            if self.checkpoint_name is not None:
                tag = f'{self.checkpoint_name}*.pt'
            else:
                tag = f'{self.__class__.__name__}-' \
                      f'{self.aux_loss.__class__.__name__}*.pt'
            valid_files = Path(checkpoint_path).rglob(tag)

            # select most recent checkpoint
            final_path = max(valid_files, key=os.path.getctime)
        else:
            raise ValueError(f'Invalid checkpoint path {path}')

        self.perturbation.load_state_dict(
            torch.load(
                final_path,
                map_location=self.pipeline.device)
        )

    def _checkpoint(self):
        """Save model/perturbation checkpoint"""
        if self.writer is not None:
            if self.checkpoint_name is not None:
                tag = f'{self.checkpoint_name}-epoch-{self._epoch_id}'
            else:
                tag = f'{self.__class__.__name__}-' \
                      f'{self.aux_loss.__class__.__name__}-' \
                      f'epoch-{self._epoch_id}'
            self.writer.checkpoint(
                self.perturbation.state_dict(),
                tag=tag,
                global_step=None
            )

    def __del__(self):
        """Save model/perturbation checkpoint upon deletion"""
        self._checkpoint()
