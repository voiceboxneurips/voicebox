import torch
import torch.nn as nn

from typing import Union

from src.models.model import Model
from src.simulation.simulation import Simulation
from src.preprocess.preprocessor import Preprocessor
from src.defenses.defense import Defense

################################################################################
# Encapsulate all stages of audio classification pipeline
################################################################################


class Pipeline(nn.Module):

    def __init__(self,
                 model: Model,
                 simulation: Simulation = None,
                 preprocessor: Preprocessor = None,
                 defense: Defense = None,
                 device: Union[str, torch.device] = 'cpu',
                 **kwargs
                 ):
        """
        Pipeline encompassing acoustic environment simulation, preprocessing,
        model, and defenses (purification and detection).

        :param model: the victim classifier
        :param simulation: an end-to-end differentiable acoustic simulation
        :param preprocessor: differentiable preprocessing stages
        :param defense: a set of purification and/or detection defenses.
                        Purification defenses are applied to incoming audio in
                        sequence, while detection defenses are applied in
                        parallel
        :param device: store device to ensure all pipeline components are
                       correctly assigned
        """
        super().__init__()

        self.model = model
        self.simulation = simulation
        self.preprocessor = preprocessor
        self.defense = defense
        self.device = device

        # flags to selectively enable pipeline stages
        self._enable_simulation = True
        self._enable_preprocessor = True
        self._enable_defense = True

        # ensure model is in 'eval' mode
        self.model.eval()

        # move all submodules to stored device
        self.set_device(device)

        # freeze gradient computation for all stored parameters
        self._freeze_grad()

        # randomly initialize simulation parameters
        self.sample_params()

    @property
    def enable_simulation(self):
        return self._enable_simulation

    @enable_simulation.setter
    def enable_simulation(self, flag: bool):
        self._enable_simulation = flag

    @property
    def enable_preprocessor(self):
        return self._enable_preprocessor

    @enable_preprocessor.setter
    def enable_preprocessor(self, flag: bool):
        self._enable_preprocessor = flag

    @property
    def enable_defense(self):
        return self._enable_defense

    @enable_defense.setter
    def enable_defense(self, flag: bool):
        self._enable_defense = flag

    def set_device(self, device: Union[str, torch.device]):
        """
        Move all submodules to stored device
        """
        self.device = device

        for module in self.modules():
            module.to(self.device)

    def _freeze_grad(self):
        """
        Disable gradient computations for all stored parameters
        """
        for p in self.parameters():
            p.requires_grad = False

    def sample_params(self):
        """
        Randomly re-sample the parameters of each stored effect
        """
        if self.simulation is not None:
            self.simulation.sample_params()

    def simulate(self, x: torch.Tensor):
        """
        Pass inputs through simulation
        """
        if self.enable_simulation and self.simulation is not None:
            x = self.simulation(x)

        return x

    def preprocess(self, x: torch.Tensor):
        """
        Pass inputs through preprocessing
        """
        if self.enable_preprocessor and self.preprocessor is not None:
            x = self.preprocessor(x)

        return x

    def purify(self, x: torch.Tensor):
        """
        Pass inputs through purification defenses
        """
        if self.enable_defense and self.defense is not None:
            x = self.defense.purify(x)

        return x

    def forward(self, x: torch.Tensor):
        """
        Pass inputs through simulation, preprocessor, purification defenses, and
        model in sequence
        """
        x = self.simulate(x)
        x = self.preprocess(x)
        x = self.purify(x)

        return self.model(x)

    def detect(self, x: torch.Tensor):
        """
        Apply detection defenses to input in parallel. For every input, each
        detection defense produces a score indicating confidence in its
        adversarial nature and a boolean flag indicating whether this score
        surpasses a (calibrated) internal threshold.

        :param x: input tensor (n_batch, ...)
        :return: flags (n_batch, n_defenses), scores (n_batch, n_defenses)
        """

        # apply simulated distortions and preprocessing to input; omit
        # purification defenses
        x = self.simulate(x)
        x = self.preprocess(x)

        if self._enable_defense and self.defense is not None:
            flags, scores = self.defense.detect(
                x,
                self.model
            )

        else:

            n_batch = x.shape[0]

            # allow zero-gradients to propagate
            flags = x.reshape(n_batch, -1).sum(dim=-1).reshape(n_batch, 1) * 0
            scores = x.reshape(n_batch, -1).sum(dim=-1).reshape(n_batch, 1) * 0

        return flags, scores

    def match_predict(self, y_pred: torch.tensor, y_true: torch.Tensor):
        """
        Determine whether target pairs are equivalent under stored model
        """
        return self.model.match_predict(y_pred, y_true)



