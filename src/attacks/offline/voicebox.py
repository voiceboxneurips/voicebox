import torch

from src.attacks.offline.perturbation import Perturbation
from src.attacks.offline.trainable import TrainableAttack
from src.attacks.offline.perturbation import VoiceBox
from src.loss.auxiliary import AuxiliaryLoss

from typing import Union

################################################################################
# VoiceBox online filtering-based attack
################################################################################


class VoiceBoxAttack(TrainableAttack):

    def __init__(self,
                 voicebox_kwargs: dict,
                 control_loss: AuxiliaryLoss = None,
                 **kwargs):

        # additional (optional) auxiliary loss on filter controls
        self.control_loss = control_loss

        super().__init__(
            perturbation=VoiceBox(**voicebox_kwargs),
            **kwargs)

    def _log_step(self,
                  x: torch.Tensor,
                  x_adv: torch.Tensor,
                  y: torch.Tensor,
                  adv_loss: Union[float, torch.Tensor] = None,
                  det_loss: Union[float, torch.Tensor] = None,
                  aux_loss: Union[float, torch.Tensor] = None,
                  control_loss: Union[float, torch.Tensor] = None,
                  success_rate: Union[float, torch.Tensor] = None,
                  detection_rate: Union[float, torch.Tensor] = None,
                  idx: int = 0,
                  tag: str = None,
                  *args,
                  **kwargs
                  ):

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

        # log control-signal loss
        self.writer.log_scalar(
            control_loss,
            f"{tag}/control-signal-loss",
            global_step=self._iter_id
        )

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
            perturbed = self.perturbation(x, y=y, *args, **kwargs)
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

            # obtain filter controls for given inputs
            get_controls = callable(
                getattr(self.perturbation, "get_controls", None))
            if self.control_loss is not None and get_controls:

                # compute slowness / sparsity loss on control signal
                controls = self.perturbation.get_controls(
                    x, *args, **kwargs)
                control_scores = self.control_loss(controls)
                control_loss = torch.mean(control_scores) * 0.01

                # backpropagate
                control_loss.backward()

                # retrieve parameter gradients
                control_loss_grad = self._retrieve_parameter_gradients(
                    self.perturbation
                ).detach()

                # add to aux loss
                aux_loss_grad = aux_loss_grad + control_loss_grad

            else:
                control_loss = 0.0
            ################################################################

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
                if self.k and self._batch_id % self.k:
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
                    control_loss=control_loss,
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

    def _evaluate_batch(self,
                        x: torch.Tensor,
                        y: torch.Tensor,
                        *args,
                        **kwargs
                        ):
        """Evaluate batch of inputs by passing through model/perturbation"""

        x_orig = x.clone().detach()
        x_adv = self.perturbation(x_orig, y=y, *args, **kwargs)
        return x_adv
