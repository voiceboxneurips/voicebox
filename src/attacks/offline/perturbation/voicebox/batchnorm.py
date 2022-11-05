import torch
import torch.nn as nn

################################################################################
# Time-distributed batch normalization layer
################################################################################


class BatchNorm(nn.Module):
    """Apply batch normalization along feature dimension only"""
    def __init__(self,
                 num_features,
                 feature_dim: int = -1, **kwargs):

        super().__init__()

        if feature_dim == 1:
            self.permute = (0, 1, 2)
        elif feature_dim in [2, -1]:
            self.permute = (0, 2, 1)
        else:
            raise ValueError(f'Must provide batch-first inputs')

        self.num_features = num_features
        self.feature_dim = feature_dim

        # pass any additional arguments to batch normalization module
        self.bn = nn.BatchNorm1d(num_features=self.num_features, **kwargs)

    def forward(self, x: torch.Tensor):

        # check input dimensions
        assert x.ndim == 3
        assert x.shape[self.feature_dim] == self.num_features

        # reshape to ensure batch normalization is time-distributed
        x = x.permute(*self.permute)

        # apply normalization
        x = self.bn(x)

        # restore original shape
        x = x.permute(*self.permute)

        return x
