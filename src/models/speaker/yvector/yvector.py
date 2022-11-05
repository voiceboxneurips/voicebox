import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.speaker.yvector.tdnn import TDNNLayer
from src.models.speaker.yvector.wav2spk import (
    ConvFeatureExtractionModel, Fp32GroupNorm, norm_block
)
import numpy as np

################################################################################
# Y-Vector implementation of Zhu et al. (2021)
################################################################################


class SEBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fgate = nn.Sequential(nn.Linear(channels, channels), nn.Sigmoid())
        self.tgate = nn.Sequential(nn.Linear(channels, 1), nn.Sigmoid())

    def forward(self, x):

        fg = self.fgate(x.mean(dim=-1))
        x = x * fg.unsqueeze(-1)

        tg = x.permute(0, 2, 1).contiguous().view(-1, x.shape[1])
        tg = self.tgate(tg).view(x.shape[0], x.shape[2]).unsqueeze(1)
        out = x * tg

        return out


class MultiScaleConvFeatureExtractionModel(nn.Module):
    def __init__(
            self,
            dropout=0.0,
            non_affine_group_norm=False,
            activation=nn.ReLU(),):
        super().__init__()

        def block(n_in, n_out, k, stride, padding=0):
            return nn.Sequential(
                nn.Conv1d(n_in, n_out, k, stride=stride, bias=False, padding=padding),
                nn.Dropout(p=dropout),
                norm_block(is_layer_norm=False, dim=n_out, affine=not non_affine_group_norm,
                           is_instance_norm=True),
                activation)

        self.conv_front = nn.ModuleList()

        # multi-3: s=18
        self.conv_front.append(nn.Sequential(block(1, 90, 36, 18, 0), block(90, 192, 5, 1, 2)))
        self.conv_front.append(nn.Sequential(block(1, 90, 18, 9, 0), block(90, 160, 5, 2, 0)))
        self.conv_front.append(nn.Sequential(block(1, 90, 12, 6, 0), block(90, 160, 5, 3, 0)))

        self.skip1 = nn.MaxPool1d(kernel_size=5, stride=8)
        self.skip2 = nn.MaxPool1d(kernel_size=3, stride=4, padding=1)
        #         self.skip3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.conv1 = block(512, 512, 5, 2)
        self.conv2 = block(512, 512, 3, 2)
        self.conv3 = block(512, 512, 3, 2, padding=2)

        self.am1 = SEBlock(512)
        self.am2 = SEBlock(512)
        self.am3 = SEBlock(512)
        self.am4 = SEBlock(512*3)

    def forward(self, x):
        # BxT -> BxCxT

        # wave encoder
        enc = []
        ft_shape = []
        for conv in self.conv_front:
            enc.append(conv(x))
            ft_shape.append(conv(x).shape[-1])

        ft_max = np.min(np.array(ft_shape))
        enc = torch.cat((enc[0][:, :, :ft_max], enc[1][:, :, :ft_max], enc[2][:, :, :ft_max]), dim=1)

        # skipping layers
        skip1_out = self.skip1(enc)
        out = self.conv1(enc)
        out = self.am1(out)
        skip2_out = self.skip2(out)
        out = self.conv2(out)
        out = self.am2(out)
        #         skip3_out = self.skip3(out)
        out = self.conv3(out)
        out = self.am3(out)

        t_max = np.min(np.array([skip1_out.shape[-1], skip2_out.shape[-1], out.shape[-1]]))

        out = torch.cat((skip1_out[:, :, :t_max], skip2_out[:, :, :t_max], out[:, :, :t_max]), dim=1)
        out = self.am4(out)

        return out


class TDNN_Block(nn.Module):
    def __init__(self, input_dim, output_dim=512, context_size=5, dilation=1, norm='bn', affine=True):
        super(TDNN_Block, self).__init__()
        if norm == 'bn':
            norm_layer = nn.BatchNorm1d(output_dim, affine=affine)
        elif norm == 'ln':
            #             norm_layer = nn.GroupNorm(1, output_dim, affine=affine)
            norm_layer = Fp32GroupNorm(1, output_dim, affine=affine)
        elif norm == 'in':
            norm_layer = nn.GroupNorm(output_dim, output_dim, affine=False)
        else:
            raise ValueError('Norm should be {bn, ln, in}.')
        self.tdnn_layer = nn.Sequential(
            TDNNLayer(input_dim, output_dim, context_size, dilation),
            norm_layer,
            nn.ReLU()
        )

    def forward(self, x):
        return self.tdnn_layer(x)


class xvecTDNN(nn.Module):
    def __init__(self, feature_dim=512, embed_dim=512, norm='bn', p_dropout=0.0):
        super(xvecTDNN, self).__init__()
        self.tdnn = nn.Sequential(
            TDNN_Block(feature_dim, 512, 5, 1, norm=norm),
            TDNN_Block(512, 512, 3, 2, norm=norm),
            TDNN_Block(512, 512, 3, 3, norm=norm),
            TDNN_Block(512, 512, 1, 1, norm=norm),
            TDNN_Block(512, 1500, 1, 1, norm=norm),
        )

        self.fc1 = nn.Linear(3000, 512)
        self.bn = nn.BatchNorm1d(512)
        self.dropout_fc1 = nn.Dropout(p=p_dropout)
        self.lrelu = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(512, embed_dim)

    def forward(self, x):
        # Note: x must be (batch_size, feat_dim, chunk_len)
        x = self.tdnn(x)

        stats = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1)

        x = self.dropout_fc1(self.lrelu(self.bn(self.fc1(stats))))
        x = self.fc2(x)

        return x


class YVector(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()

        self.feature_encoder = MultiScaleConvFeatureExtractionModel()
        self.tdnn_aggregator = xvecTDNN(feature_dim=512*3, embed_dim=128, norm='ln')

    def forward(self, x):

        # require batch and channel dimensions
        assert x.ndim >= 2

        # avoid modifying input audio
        n_batch, *channel_dims, signal_len = x.shape

        # add channel dimension if necessary
        if len(channel_dims) == 0:
            x = x.unsqueeze(1)

        out = self.feature_encoder(x)
        out = self.tdnn_aggregator(out)

        return out


if __name__ == "__main__":

    model = YVector()
    print(model)
    wav_input_16khz = torch.randn(4, 1, 48000)
    c = model(wav_input_16khz)
    print(c.shape)
