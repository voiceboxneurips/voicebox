import torch
import torch.nn as nn
import torch.nn.functional as F

################################################################################
# Post-bottleneck lookahead module from DeepSpeech 2
################################################################################


class Lookahead(nn.Module):
    """
    Lookahead Convolution Layer for Unidirectional Recurrent Neural Networks
    from Wang et al 2016.
    """
    def __init__(self, n_features: int, lookahead_frames: int):
        """
        Parameters
        ----------
        n_features (int):       feature dimension
        lookahead_frames (int): lookahead length in frames
        """
        super(Lookahead, self).__init__()

        assert lookahead_frames >= 0, 'Must provide nonzero context length'

        self.lookahead_frames = lookahead_frames
        self.n_features = n_features

        # pad to preserve sequence length in output
        self.pad = (0, self.lookahead_frames)

        self.conv = nn.Conv1d(
            self.n_features,
            self.n_features,
            kernel_size=self.lookahead_frames + 1,
            stride=1,
            groups=self.n_features,  # independence between features
            padding=0,
            bias=False
        )

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x (Tensor): shape (n_batch, n_frames, n_features)

        Returns
        -------
        out (Tensor): shape (n_batch, n_frames, n_features)
        """
        x = x.transpose(1, 2)  # (n_batch, n_features, n_frames)
        x = F.pad(x, pad=self.pad, value=0)
        x = self.conv(x)
        x = x.transpose(1, 2).contiguous()  # (n_batch, n_features, n_frames)
        return x
