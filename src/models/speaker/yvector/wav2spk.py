import math
import torch
import torch.nn as nn
import torch.nn.functional as F

################################################################################
# Wav2Spk implementation of Zhu et al. (2021)
################################################################################


class Fp32GroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, inputs):
        output = F.group_norm(
            inputs.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(inputs)


class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, inputs):
        output = F.layer_norm(
            inputs.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(inputs)


class TransposeLast(nn.Module):
    def __init__(self, deconstruct_idx=None):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(-2, -1)


def norm_block(is_layer_norm, dim, affine=True, is_instance_norm=False):
    if is_layer_norm:
        mod = nn.Sequential(
            TransposeLast(),
            Fp32LayerNorm(dim, elementwise_affine=affine),
            TransposeLast(),
        )
    else:
        if is_instance_norm:
            mod = Fp32GroupNorm(dim, dim, affine=False)  # instance norm
        else:
            mod = Fp32GroupNorm(1, dim, affine=affine)  # layer norm

    return mod


class ConvFeatureExtractionModel(nn.Module):
    def __init__(
            self,
            conv_layers,
            dropout=0.0,
            log_compression=False,
            skip_connections=False,
            residual_scale=0.5,
            non_affine_group_norm=False,
            activation=nn.ReLU(),
            is_instance_norm=True,
    ):
        super().__init__()

        def block(n_in, n_out, k, stride):
            return nn.Sequential(
                nn.Conv1d(n_in, n_out, k, stride=stride, bias=False),
                nn.Dropout(p=dropout),
                norm_block(is_layer_norm=False,
                           dim=n_out,
                           affine=not non_affine_group_norm,
                           is_instance_norm=is_instance_norm),
                activation,
            )

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for dim, k, stride in conv_layers:
            self.conv_layers.append(block(in_d, dim, k, stride))
            in_d = dim

        self.log_compression = log_compression
        self.skip_connections = skip_connections
        self.residual_scale = math.sqrt(residual_scale)

    def forward(self, x):
        # BxT -> BxCxT
        # x = x.unsqueeze(1)

        for conv in self.conv_layers:
            residual = x
            x = conv(x)
            if self.skip_connections and x.size(1) == residual.size(1):
                tsz = x.size(2)
                r_tsz = residual.size(2)
                residual = residual[..., :: r_tsz // tsz][..., :tsz]
                x = (x + residual) * self.residual_scale

        if self.log_compression:
            x = x.abs()
            x = x + 1
            x = x.log()

        return x


class ZeroPad1d(nn.Module):
    def __init__(self, pad_left, pad_right):
        super().__init__()
        self.pad_left = pad_left
        self.pad_right = pad_right

    def forward(self, x):
        return F.pad(x, (self.pad_left, self.pad_right))


class ConvAggegator(nn.Module):
    def __init__(
            self,
            conv_layers,
            embed,
            dropout=0.0,
            skip_connections=False,
            residual_scale=0.5,
            non_affine_group_norm=False,
            conv_bias=True,
            zero_pad=False,
            activation=nn.ReLU(),
    ):
        super().__init__()

        def block(n_in, n_out, k, stride):
            # padding dims only really make sense for stride = 1
            ka = k // 2
            kb = ka - 1 if k % 2 == 0 else ka

            pad = (
                ZeroPad1d(ka + kb, 0) if zero_pad else
                nn.ReplicationPad1d((ka + kb, 0))
            )

            return nn.Sequential(
                pad,
                nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias),
                nn.Dropout(p=dropout),
                norm_block(False, n_out, affine=not non_affine_group_norm),
                activation,
            )

        in_d = embed
        self.conv_layers = nn.ModuleList()
        self.residual_proj = nn.ModuleList()
        for dim, k, stride in conv_layers:
            if in_d != dim and skip_connections:
                self.residual_proj.append(nn.Conv1d(in_d, dim, 1, bias=False))
            else:
                self.residual_proj.append(None)

            self.conv_layers.append(block(in_d, dim, k, stride))
            in_d = dim
        self.conv_layers = nn.Sequential(*self.conv_layers)
        self.skip_connections = skip_connections
        self.residual_scale = math.sqrt(residual_scale)

    def forward(self, x):
        for rproj, conv in zip(self.residual_proj, self.conv_layers):
            residual = x
            x = conv(x)
            if self.skip_connections:
                if rproj is not None:
                    residual = rproj(residual)
                x = (x + residual) * self.residual_scale
        return x


class StatsPooling(nn.Module):
    def __init__(self):
        super(StatsPooling,self).__init__()

    def forward(self, varient_length_tensor):
        mean = varient_length_tensor.mean(dim=-1)
        std = varient_length_tensor.std(dim=-1)
        return torch.cat((mean,std),dim=1)


class architecture(nn.Module):
    def __init__(self,
                 feature_enc_layers=[(40, 10, 5), (200, 5, 4), (300, 5, 2)] +
                                    [(512, 3, 2)]*2,
                 agg_layers=[(512, 3, 1)] * 4):
        super(architecture, self).__init__()
        #         self.ln = nn.GroupNorm(1, 1, eps=1e-8)

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            is_instance_norm=True)
        self.temporal_gating = nn.Sequential(
            nn.Linear(feature_enc_layers[-1][0], 1),
            nn.Sigmoid())
        self.feature_aggregator = ConvAggegator(
            conv_layers=agg_layers,
            embed=feature_enc_layers[-1][0])

        self.statspool = StatsPooling()
        self.fc1 = nn.Linear(agg_layers[-1][0] * 2, 512)
        self.bn = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 128)

        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        '''
           x: [B, L]
        '''

        if isinstance(self.feature_extractor, nn.modules.container.ModuleList):
            z = []
            for encoder in self.feature_extractor:
                z.append(encoder(x))
            z = torch.cat(z, dim=1)
        else:
            z = self.feature_extractor(x)

        # Temporal gating
        g = z.permute(0, 2, 1).contiguous().view(-1, z.shape[1])
        g = self.temporal_gating(g).view(z.shape[0], z.shape[2]).unsqueeze(1)
        z = z * g

        c = self.feature_aggregator(z)
        c = self.statspool(c)

        # To use the regularization on the last two layers for fair comparison, may differ from original wav2spk
        x = self.lrelu(self.bn(self.fc1(c)))
        x = self.fc2(x)

        return x
