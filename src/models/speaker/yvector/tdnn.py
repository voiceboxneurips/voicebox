import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

################################################################################
# TDNN implementation of Zhu et al. (2021)
################################################################################


class TDNNLayer(nn.Module):

    def __init__(self, input_dim, output_dim,
                 context_size, dilation=1):
        '''
        TDNN as defined by https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf

        Affine transformation not applied globally to all frames but smaller windows with local context

        batch_norm: True to include batch normalisation after the non linearity

        Context size and dilation determine the frames selected
        (although context size is not really defined in the traditional sense)
        For example:
            context size 5 and dilation 1 is equivalent to [-2,-1,0,1,2]
            context size 3 and dilation 2 is equivalent to [-2, 0, 2]
            context size 1 and dilation 1 is equivalent to [0]
        '''
        super(TDNNLayer, self).__init__()
        self.context_size = context_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.kernel = nn.Linear(input_dim*context_size, output_dim)

    def forward(self, inputs):
        '''
        input: size (batch, input_features, seq_len)
        outpu: size (batch, new_seq_len, output_features)
        '''

        # ----------Convolution = unfold + matmul + fold
        x = inputs
        _, d, _ = x.shape
        assert (d == self.input_dim), 'Input dimension was wrong. Expected ({}), got ({})'.format(self.input_dim, d)
        x = x.unsqueeze(1)

        # Unfold input into smaller temporal contexts
        x = F.unfold(x, (self.input_dim, self.context_size),
                     stride=(self.input_dim, 1),
                     dilation=(1, self.dilation))

        # N, output_dim*context_size, new_t = x.shape
        x = x.transpose(1, 2)
        x = self.kernel(x) # matmul

        # transpose to channel first
        x = x.transpose(1, 2)

        return x


class TDNNBlock(nn.Module):

    def __init__(self, input_dim, bn_dim,
                 skip, context_size, dilation=1,
                 bottleneck=False):
        '''
        TDNNBlock
        '''
        super(TDNNBlock, self).__init__()

        # bn conv
        self.bottleneck = bottleneck
        if bottleneck:
            self.bnconv1d = nn.Conv1d(input_dim, bn_dim, 1)
            self.nonlinear1 = nn.PReLU()
            self.norm1 = nn.GroupNorm(1, bn_dim, eps=1e-08)
            self.tdnnblock = TDNNLayer(bn_dim, input_dim, context_size, dilation)
        else:
            self.tdnnblock = TDNNLayer(input_dim, input_dim, context_size, dilation)

        # tdnn
        self.nonlinear2 = nn.PReLU()
        self.norm2 = nn.GroupNorm(1, input_dim, eps=1e-08)

        # skip connection
        self.skip = skip
        if self.skip:
            self.skip_out = nn.MaxPool1d(kernel_size=context_size,
                                         stride=1, dilation=dilation)

    def forward(self, x):
        '''
        input: size (batch, seq_len, input_features)
        outpu: size (batch, new_seq_len, output_features)
        '''
        out = x
        if self.bottleneck:
            out = self.nonlinear1(self.bnconv1d(out))
            out = self.norm1(out)

        out = self.nonlinear2(self.tdnnblock(out))
        out = self.norm2(out)

        if self.skip:
            skip = self.skip_out(x)
            return out, skip
        else:
            return out


class TDNN(nn.Module):

    def __init__(self, filter_dim, input_dim, bn_dim,
                 skip, context_size=3, layer=9, stack=1,
                 bottleneck=False):
        '''
        stacked TDNN Blocks
        '''
        super(TDNN, self).__init__()

        #         # BottleNeck Layer
        #         self.LN = nn.GroupNorm(1, filter_dim, eps=1e-8)
        #         self.BN_conv = nn.Conv1d(filter_dim, input_dim, 1)

        # Residual Connection
        self.skip = skip

        # TDNN for feature extraction
        self.receptive_field = 0

        self.tdnn = nn.ModuleList([])
        for s in range(stack):
            for i in range(layer):
                self.tdnn.append(TDNNBlock(input_dim, bn_dim, self.skip,
                                           context_size=3, dilation=2**i,
                                           bottleneck=bottleneck))

            if i == 0 and s == 0:
                self.receptive_field += context_size
            else:
                self.receptive_field += (context_size - 1) * 2 ** i

        print("Receptive field: {:3d} frames.".format(self.receptive_field))


    def forward(self, x):
        '''
        input: size (batch, seq_len, input_features)
        outpu: size (batch, new_seq_len, output_features)
        '''

        #         output = self.BN_conv(self.LN(x))

        for i in range(len(self.tdnn)):
            if self.skip:
                output, skips = self.tdnn[i](x)
                output = skips + output
            else:
                output = self.tdnn[i](output)

        return output
