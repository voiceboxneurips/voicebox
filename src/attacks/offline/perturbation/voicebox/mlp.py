import torch
import torch.nn as nn

################################################################################
# Time-distributed MLP module
################################################################################


class MLP(nn.Module):
    """Time-distributed MLP network"""
    def __init__(self,
                 in_channels: int,
                 hidden_size: int = 512,
                 depth: int = 2,
                 activation: nn.Module = nn.LeakyReLU()
                 ):

        super().__init__()

        channels = [in_channels] + depth * [hidden_size]
        mlp = []
        for i in range(depth):
            mlp.append(nn.Linear(channels[i], channels[i + 1]))
            mlp.append(nn.LayerNorm(channels[i + 1]))

            # omit nonlinearity after final layer
            if i < depth - 1:
                mlp.append(activation)

        self.mlp = nn.Sequential(*mlp)

    def forward(self, x: torch.Tensor):
        return self.mlp(x)
