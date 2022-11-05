import torch
import torch.nn as nn

from typing import Dict

################################################################################
# Base class for all adversarial perturbation operators
################################################################################


class Perturbation(nn.Module):
    """
    Base class for adversarial perturbation operators. Allows (recursive)
    composition of Perturbation objects and facilitates projected gradient
    descent optimization by controlling parameter and gradient access.

    Subclasses must override the methods `forward()`, `set_reference()`, and
    `_project_valid_top_level()`. Optionally, sublasses can overwrite
    `_visualize_top_level()` to produce parameter visualizations for logging
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def _check_broadcastable(x: torch.Tensor, x_ref: torch.Tensor):
        """
        Check whether input and reference tensors are broadcastable
        """

        broadcastable = all(
            (m == n) or (m == 1) or (n == 1) for m, n in zip(
                x.shape[::-1], x_ref.shape[::-1]
            )
        )

        # broadcast cannot expand input batch dimension
        valid = x.shape[0] == x_ref.shape[0] or x_ref.shape[0] == 1

        return broadcastable * valid

    @staticmethod
    def _freeze_grad(m: nn.Module):
        """
        Disable gradient computation for all parameters in given module
        :param m: torch.nn.Module object
        """
        for module in m.modules():
            for param in module.parameters():
                param.requires_grad = False

    def set_reference(self, x: torch.Tensor):
        """
        Given reference input, initialize perturbation parameters accordingly
        and match input device.

        :param x: reference audio, shape (n_batch, n_channels, signal_length)
        """

        raise NotImplementedError()

    def forward(self, x: torch.Tensor, *args, **kwargs):
        """
        Apply perturbation to inputs.

        :param x: input audio, shape (n_batch, n_channels, signal_length)
        """
        raise NotImplementedError()

    def retrieve_parameter_gradients(self):
        """
        Retrieve 'flattened' perturbation parameter representation, including
        those of all stored Perturbation objects.

        return: flattened parameter gradients, shape  (n_parameters,)
        """

        flattened_grad = []

        for param in self.parameters():
            if param.requires_grad:

                if param.grad is None:
                    flattened_grad.append(
                        torch.zeros_like(param).detach().flatten()
                    )
                else:
                    flattened_grad.append(param.grad.detach().flatten())

        return torch.cat(flattened_grad, dim=-1)

    def set_parameter_gradients(self, flattened_grad: torch.tensor):
        """
        Given flattened gradients, apply to stored parameters.

        :param flattened_grad: parameter gradients of shape (n_parameters,)
        """

        # check that flattened gradients have valid shape
        prod = sum(
            [p.shape.numel() for p in self.parameters() if p.requires_grad]
        )

        assert flattened_grad.ndim <= 1
        assert flattened_grad.numel() == prod

        idx = 0
        for param in self.parameters():
            if param.requires_grad:
                param_length = param.shape.numel()
                grad = flattened_grad[idx:idx + param_length].reshape(
                    param.shape
                ).detach()
                param.grad = grad
                idx += param_length

    def _visualize_top_level(self) -> Dict[str, torch.Tensor]:
        """
        Visualize top-level (non-recursive) perturbation parameters.

        :return: tag (string) / image (tensor) pairs, stored in a dictionary
        """
        return {}

    def visualize(self) -> Dict[str, torch.Tensor]:
        """
        Visualize perturbation parameters.

        :return: tag (string) / image (tensor) pairs, stored in a dictionary
        """

        # collect visualizations for top-level parameters
        visualizations = self._visualize_top_level()

        # collect visualizations for stored Perturbation objects
        for m in self.children():
            if isinstance(m, Perturbation):
                visualizations = {**visualizations, **m.visualize()}

        return visualizations

    def _project_valid_top_level(self):
        """
        Project top-level (non-recursive) parameters to valid range.
        """
        raise NotImplementedError()

    def project_valid(self):
        """
        Project perturbation parameters to valid range. Apply to all stored
        Perturbation objects recursively, such that each Perturbation is
        responsible for its own projection.
        """

        # project top-level parameters (non-recursive)
        self._project_valid_top_level()

        for m in self.children():
            if isinstance(m, Perturbation):
                m.project_valid()
