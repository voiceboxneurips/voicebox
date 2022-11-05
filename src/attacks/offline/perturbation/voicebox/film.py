import torch
import torch.nn as nn

from src.attacks.offline.perturbation.voicebox.batchnorm import BatchNorm

################################################################################
# Affine conditioning layer
################################################################################


class FiLM(nn.Module):
    """
    Affine conditioning layer, as proposed in Perez et al.
    (https://arxiv.org/pdf/1709.07871.pdf). Operates on each channel of a
    selected feature representation, with one learned scaling parameter and one
    learned bias parameter per channel.

    Code adapted from https://github.com/csteinmetz1/steerable-nafx
    """
    def __init__(
            self,
            cond_dim: int,
            num_features: int,
            batch_norm: bool = True,
    ):
        """
        Apply linear projection and batch normalization to obtain affine
        conditioning parameters.

        :param cond_dim: dimension of conditioning input
        :param num_features: number of feature maps to which conditioning is
                             applied
        :param batch_norm: if True, perform batch normalization
        """
        super().__init__()
        self.num_features = num_features
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = BatchNorm(num_features, feature_dim=-1, affine=False)
        self.adaptor = torch.nn.Linear(cond_dim, num_features * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """

        FIGURE OUT SHAPES

        x (Tensor):    shape
        cond (Tensor): shape
        """

        # linear projection of conditioning input
        cond = self.adaptor(cond)

        # learn scale and bias parameters per channel, thus 2X num_features
        g, b = torch.chunk(cond, 2, dim=-1)

        if self.batch_norm:
            x = self.bn(x)  # apply BatchNorm without affine
        x = (x * g) + b  # then apply conditional affine

        return x
