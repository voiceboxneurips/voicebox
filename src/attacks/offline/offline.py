import torch
import torch.nn as nn

from typing import Tuple, Union
from torch.utils.data import Dataset, TensorDataset, DataLoader

from src.utils.writer import Writer
from src.pipelines.pipeline import Pipeline
from src.loss.adversarial import AdversarialLoss
from src.loss.auxiliary import AuxiliaryLoss

################################################################################
# Base class for offline adversarial attacks
################################################################################


class OfflineAttack:

    def __init__(self,
                 pipeline: Pipeline,
                 adv_loss: AdversarialLoss,
                 aux_loss: AuxiliaryLoss = None,
                 batch_size: int = 32,
                 rand_evals: int = 0,
                 writer: Writer = None,
                 **kwargs
                 ):
        """
        Base class for offline attacks. Subclasses must override the
        `evaluate_batch()` method.

        Offline attacks optimize perturbations of benign inputs without
        real-time performance constraints. Optimization is performed using a
        stored Pipeline object, encompassing a victim model, acoustic
        simulation, and adversarial defenses.

        :param pipeline: a Pipeline object wrapping a (defended) classifier
        :param adv_loss: AdversarialLoss object encapsulating attacker objective
        :param aux_loss: optional AuxiliaryLoss object encapsulating
                         some perceptibility objective
        :param batch_size: batch size for attack
        :param rand_evals: randomly-resampled simulated evaluations per each
                           final generated attack
        """
        self.pipeline = pipeline

        # ensure gradients flow through PyTorch RNN layers
        self._pipeline_rnn_grad()

        self.adv_loss = adv_loss
        self.aux_loss = aux_loss

        self.batch_size = batch_size
        self.rand_evals = rand_evals

        # log attack progress
        self.writer = writer

        # track batch inputs
        self._batch_id = 0
        self._iter_id = 0

        # optional data-loading arguments
        self.pin_memory = kwargs.get('pin_memory', False)
        self.num_workers = kwargs.get('num_workers', 0)

        self._check_loss()

    def _pipeline_rnn_grad(self):
        """
        PyTorch requires any recurrent modules be placed in `train` mode to
        enable backpropagation through the pipeline.
        """
        for m in self.pipeline.modules():
            if isinstance(m, nn.RNNBase):
                m.train()

    def _check_loss(self):
        """
        Validate adversarial and auxiliary losses
        """

        assert self.adv_loss is not None, 'Must provide adversarial loss'
        assert self.adv_loss.reduction in ['none', None], \
            'All losses must provide unreduced scores'

        assert self.aux_loss is None or \
               self.aux_loss.reduction in ['none', None], \
            'All losses must provide unreduced scores'

    @staticmethod
    def _create_dataset(x: torch.Tensor, y: torch.Tensor):
        """
        If attack inputs are given as tensors, create a simple dataset
        """

        # require batch dimension
        assert x.ndim >= 2

        dataset = TensorDataset(
            x.type(torch.float32),
            y.type(torch.float32),
        )
        return dataset

    def _compute_detection_array(self, x_adv, *args, **kwargs):
        """
        Pass attack audio through any detection defenses in stored Pipeline, and
        return boolean detection flags for each input
        """
        flags, scores = self.pipeline.detect(x_adv)
        return flags

    @torch.no_grad()
    def _compute_success_array(self,
                               x: torch.Tensor,
                               y: torch.Tensor,
                               x_adv: torch.Tensor,
                               *args,
                               **kwargs
                               ):
        """
        Pass attack audio through stored Pipeline and determine adversarial
        success for each input
        """

        # obtain 'clean' and adversarial predictions of stored Pipeline
        preds = self.pipeline(x.detach())
        adv_preds = self.pipeline(x_adv.detach())

        # for a targeted attack, attempt to match given targets
        if self.adv_loss.targeted:
            attack_success = self.pipeline.match_predict(adv_preds, y)

        # for an untargeted attack, attempt to evade clean predictions
        else:
            attack_success = ~self.pipeline.match_predict(adv_preds, preds)

        return attack_success

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

        :param x: batch of original inputs
        :param x_adv: batch of adversarial inputs
        :param y: batch of adversarial targets
        :param adv_loss: adversarial loss value
        :param det_loss: detection loss value
        :param aux_loss: auxiliary loss value
        :param success_rate: adversarial success rate
        :param detection_rate: adversarial defense detection rate
        :param idx: batch index for logging individual examples
        """

        if self.writer is None or self._iter_id % self.writer.log_iter:
            return

        if tag is None:
            tag = f'{self.__class__.__name__}-batch-{self._batch_id}'

        x = x.clone().detach()
        x_adv = x_adv.clone().detach()

        # compute losses and simulated audio
        with torch.no_grad():
            outputs_adv = self.pipeline(x_adv)
            simulated = self.pipeline.simulate(x)
            simulated_adv = self.pipeline.simulate(x_adv)

            # if adversarial loss is not provided, compute
            if adv_loss is None:
                adv_loss = self.adv_loss(outputs_adv, y).mean()

            # if detector loss or rate is not provided, compute
            if det_loss is None or detection_rate is None:
                flags, scores = self.pipeline.detect(x_adv)
                det_loss = scores.mean()
                detection_rate = torch.mean(1.0 * flags)

            # if auxiliary loss is not provided, compute
            if aux_loss is None:
                aux_loss = 0.0 if self.aux_loss is None else self.aux_loss(
                    x_adv, x
                ).mean()

            # if adversarial success rate is not provided, compute
            if success_rate is None:
                success = self._compute_success_array(
                    x=x,
                    x_adv=x_adv,
                    y=y
                )
                success_rate = torch.mean(1.0 * success)

        # unperturbed input
        self.writer.log_audio(
            x[idx],
            f"{tag}/original",
            global_step=self._iter_id
        )

        # simulated unperturbed input
        self.writer.log_audio(
            simulated[idx],
            f"{tag}/simulated-original",
            global_step=self._iter_id
        )

        # adversarial input
        self.writer.log_audio(
            x_adv[idx],
            f"{tag}/adversarial",
            global_step=self._iter_id
        )

        # simulated adversarial input
        self.writer.log_audio(
            simulated_adv[idx],
            f"{tag}/simulated-adversarial",
            global_step=self._iter_id
        )

        # adversarial loss value
        self.writer.log_scalar(
            adv_loss,
            f"{tag}/adversarial-loss",
            global_step=self._iter_id
        )

        # detector loss value
        self.writer.log_scalar(
            det_loss,
            f"{tag}/detector-loss",
            global_step=self._iter_id
        )

        # auxiliary loss value
        self.writer.log_scalar(
            aux_loss,
            f"{tag}/auxiliary-loss",
            global_step=self._iter_id
        )

        # adversarial success rate
        self.writer.log_scalar(
            success_rate,
            f"{tag}/success-rate",
            global_step=self._iter_id
        )

        # adversarial detection rate
        self.writer.log_scalar(
            detection_rate,
            f"{tag}/detection-rate",
            global_step=self._iter_id
        )

    def _evaluate_batch(self,
                        x: torch.Tensor,
                        y: torch.Tensor,
                        **kwargs
                        ):
        """
        Perform attack on a batch of inputs.

        :param x: input tensor of shape (n_batch, ...)
        :param y: targets tensor of shape (n_batch, ...) in case of targeted
                  attack; original labels tensor of shape (n_batch, ...) in
                  case of untargeted attack
        """
        raise NotImplementedError()

    @torch.no_grad()
    def evaluate(self,
                 x: torch.Tensor = None,
                 y: torch.Tensor = None,
                 dataset: Dataset = None,
                 **kwargs
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform attack given input-target pairs, optionally in the form of a
        Dataset object. Random evaluations will then be conducted on all
        generated attacks.

        :param x: audio input, shape (n_batch, ..., signal_length)
        :param y: targets, shape (n_batch, ...)
        :param dataset: optionally, provide inputs and targets as dataset
        :return: tuple holding
                   * adversarial audio (n_batch, ..., signal_length)
                   * boolean adversarial success indicators (n_batch,)
                   * boolean adversarial detection indicators (n_batch,)
        """

        assert (x is not None and y is not None) or dataset is not None

        # prepare batched data-loading, store original device
        if dataset is None:
            orig_device = x.device
            dataset = self._create_dataset(x, y)
            x_ref = x[0:1].clone().detach()
        else:
            ref_batch = next(iter(dataset))
            if isinstance(ref_batch, tuple):
                x_ref = ref_batch[0]
            elif isinstance(ref_batch, dict):
                x_ref = ref_batch['x']
            else:
                x_ref = ref_batch
            orig_device = x_ref.device

        # prepare to compute attack success and detection rates
        attack_success = torch.zeros(
            len(dataset), dtype=torch.float).to(self.pipeline.device)
        attack_detection = torch.zeros(
            len(dataset), dtype=torch.float).to(self.pipeline.device)

        # prepare to store attack outputs
        adv_x = torch.stack(
            [torch.zeros(x_ref.shape)] * len(dataset),
            dim=0
        ).to(self.pipeline.device)

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

        # compute attacks with batching
        for (batch_id, batch_all) in enumerate(data_loader):

            self._batch_id = batch_id

            # allow for different dataset formats
            if isinstance(batch_all, tuple):
                batch_all = {
                    'x': batch_all[0],
                    'y': batch_all[1]
                }

            # match devices
            for k in batch_all.keys():
                batch_all[k] = batch_all[k].to(self.pipeline.device)

            batch_index_1 = batch_id * self.batch_size
            batch_index_2 = (batch_id + 1) * self.batch_size

            # compute attacks for given batch
            adversarial_batch = self._evaluate_batch(
                **batch_all, **kwargs
            )

            # if no random trials, evaluate once
            if not self.rand_evals:

                # compute and store success rates for batch
                attack_success_batch = self._compute_success_array(
                    **batch_all,
                    x_adv=adversarial_batch
                ).reshape(-1).type(torch.float32)

                # compute and store detection rates for batch
                attack_detection_batch = self._compute_detection_array(
                    x_adv=adversarial_batch
                ).reshape(-1).type(torch.float32)

            # otherwise, perform multiple random evaluations per attack
            else:

                # track batch success and detection over random evaluation
                success_combined_batch = []
                detection_combined_batch = []

                for i in range(self.rand_evals):

                    # randomly sample simulation parameters
                    self.pipeline.sample_params()

                    # compute and store success rates for batch
                    rand_success_batch = self._compute_success_array(
                        **batch_all,
                        x_adv=adversarial_batch
                    ).reshape(-1, 1)
                    success_combined_batch.append(rand_success_batch)

                    # compute and store detection rates for batch
                    rand_detection_batch = self._compute_detection_array(
                        x_adv=adversarial_batch
                    )
                    detection_combined_batch.append(rand_detection_batch)

                # average results over all trials
                attack_success_batch = (1.0 * torch.cat(
                    success_combined_batch, dim=-1
                )).mean(dim=-1)

                attack_detection_batch = (1.0 * torch.cat(
                    detection_combined_batch, dim=-1
                )).mean(dim=-1)

            # store generated attack audio
            adv_x[batch_index_1:batch_index_2] = adversarial_batch

            # store success rate per generated attack
            attack_success[batch_index_1:batch_index_2] = attack_success_batch

            # store detection rate per generated attack
            attack_detection[batch_index_1:batch_index_2] = attack_detection_batch

        return (adv_x.to(orig_device),
                attack_success.to(orig_device),
                attack_detection.to(orig_device))
