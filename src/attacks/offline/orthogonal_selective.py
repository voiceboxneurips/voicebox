import torch
import torch.nn as nn

################################################################################
# Mixin class for handling selective/orthogonal PGD variants
################################################################################


class SelectiveOrthogonalPGDMixin(object):

    def __init__(self, **kwargs):
        pass

    @staticmethod
    def _dot(x1: torch.Tensor, x2: torch.Tensor):
        """
        Compute batch dot product along final dimension
        """
        return (x1*x2).sum(-1, keepdim=True)

    def _project_orthogonal(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        Compute projection component of x1 along x2. For projection
        onto zero vector, return zero vector
        """
        return x2 * (self._dot(x1, x2) / self._dot(x2, x2).clamp_min(1e-12))

    def _component_orthogonal(self,
                              x1: torch.Tensor,
                              x2: torch.Tensor,
                              x3: torch.Tensor):
        """
        Compute component of x1 approximately orthogonal to x2 and x3
        """
        return x1 - self._project_orthogonal(
            x1, x2 - self._project_orthogonal(x2, x3)
        ) - self._project_orthogonal(x1, x3)

    @staticmethod
    def _retrieve_parameter_gradients(m: nn.Module):
        """
        Retrieve all trainable parameters of a nn.Module object
        :return: tensor of shape (n_parameters,)
        """

        flattened_grad = []

        for param in m.parameters():
            if param.requires_grad:
                if param.grad is None:
                    flattened_grad.append(
                        torch.zeros_like(param).detach().flatten()
                    )
                else:
                    flattened_grad.append(param.grad.detach().flatten())

        return torch.cat(flattened_grad, dim=-1)

    @staticmethod
    def _set_parameter_gradients(flattened_grad: torch.Tensor, m: nn.Module):
        """
        Set gradient attributes of trainable parameters of a nn.Module object
        :param params: tensor of shape (n_parameters,)
        """

        # check that flattened gradients have valid shape
        prod = sum(
            [p.shape.numel() for p in m.parameters() if p.requires_grad]
        )

        assert flattened_grad.ndim <= 1
        assert flattened_grad.numel() == prod

        idx = 0
        for param in m.parameters():
            if param.requires_grad:
                param_length = param.shape.numel()
                grad = flattened_grad[idx:idx + param_length].reshape(
                    param.shape
                ).detach()
                param.grad = grad
                idx += param_length
