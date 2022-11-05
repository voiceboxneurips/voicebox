import math

import torch
import torch.nn as nn
import torch.nn.functional as F

################################################################################
# Latent bottleneck modules
################################################################################


class RNNBottleneck(nn.Module):

    def __init__(self,
                 input_size: int = 512,
                 hidden_size: int = 2048,
                 proj_size: int = 512,
                 num_layers: int = 8,
                 downsample_index: int = 1,
                 downsample_factor: int = 2,
                 dropout_prob: float = 0
                 ):
        super().__init__()

        # downsampling can occur no earlier than first layer
        assert downsample_index >= 0

        self.num_layers = num_layers
        self.downsample_index = downsample_index
        self.downsample_factor = downsample_factor
        self.dropout = nn.Dropout(dropout_prob)

        # optionally, apply projection
        if proj_size >= hidden_size:
            proj_size = 0

        # build multi-layer recurrent network
        rnn = []
        for i in range(num_layers):

            # first layer
            if i == 0:
                pass

            # pre-downsampling layers
            elif i <= downsample_index:
                input_size = proj_size or hidden_size

            # immediately after downsampling layer
            elif i == downsample_index + 1:
                input_size = (proj_size or hidden_size) * downsample_factor

            # subsequent layers
            else:
                input_size = proj_size or hidden_size

            rnn.append(
                nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    proj_size=proj_size,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=False,
                    bias=True
                )
            )
        self.rnn = nn.ModuleList(rnn)

    def forward(self, x):

        for i in range(self.num_layers):

            x, _ = self.rnn[i](x, hx=None)

            x = self.dropout(x)

            if i == self.downsample_index:

                n_batch, n_frames, proj_size = x.shape

                # determine necessary padding to allow temporal downsampling
                pad_len = self.downsample_factor * math.ceil(n_frames / self.downsample_factor) - n_frames

                # apply causal padding
                x = F.pad(x, (0, 0, 0, pad_len))

                # apply temporal downsampling
                x = torch.reshape(x, (n_batch, x.shape[1] // self.downsample_factor, x.shape[2] * self.downsample_factor))

        return x


class CausalTransformer(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 dim_feedforward: int = 2048,
                 depth: int = 2,
                 heads: int = 8,
                 dropout_prob: float = 0.0):

        super().__init__()

        attention_block = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout_prob,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=attention_block,
            num_layers=depth,
            norm=None
        )

    def forward(self, x: torch.Tensor):

        _, n_frames, _ = x.shape
        causal_mask = torch.triu(torch.ones((n_frames, n_frames), dtype=torch.bool, device=x.device), diagonal=1)

        return self.transformer(x, mask=causal_mask)
