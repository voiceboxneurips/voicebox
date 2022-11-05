import math
from typing import List, Union, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import Spectrogram
from torch.cuda.amp import autocast

from src.data import DataProperties

################################################################################
# DeepSpeech2 model (Amodei et al.) as implemented by Sean Naren
################################################################################


class SequenceWise(nn.Module):

    def __init__(self, module: nn.Module):
        """
        Collapses input of shape (seq_len, n_batch, n_features) to
        (seq_len * n_batch, n_features) and applies a nn.Module along the
        feature dimension. Allows handling of variable sequence lengths and batch
        sizes.

        Parameters
        ----------
        module (nn.Module): module to apply to input
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x: torch.Tensor):

        # assume input shape (seq_len, n_batch, n_features)
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class MaskConv(nn.Module):

    def __init__(self, seq_module: nn.Sequential):
        """
        Adds padding to the output of each layer in a given convolution stack
        based on a set of given lengths. This ensures that the results of the
        model do not change when batch sizes change during inference. Expects
        input with shape (n_batch, n_channels, ???, seq_len)

        Parameters
        ----------
        seq_module (nn.Sequential): the sequential module containing the
                                    convolution stack
        """
        super(MaskConv, self).__init__()
        self.seq_module = seq_module

    def forward(self, x: torch.Tensor, lengths: Iterable):
        """

        Parameters
        ----------
        x (Tensor):     input with shape (n_batch, n_channels, ???, seq_len)
        lengths (list): list of target lengths

        Returns
        -------
        masked (Tensor): padded output of convolution stack
        lengths (list):  list of target lengths
        """
        for module in self.seq_module:
            x = module(x)
            mask = torch.BoolTensor(x.size()).fill_(0)
            if x.is_cuda:
                mask = mask.cuda()
            for i, length in enumerate(lengths):
                length = length.item()
                if (mask[i].size(2) - length) > 0:
                    mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
            x = x.masked_fill(mask, 0)
        return x, lengths


class InferenceBatchSoftmax(nn.Module):
    """Apply softmax along final tensor dimension in inference mode only"""
    def forward(self, input_: torch.Tensor):
        if not self.training:
            return F.softmax(input_, dim=-1)
        else:
            return input_


class BatchRNN(nn.Module):
    """RNN layer with optional batch normalization"""
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 rnn_type=nn.LSTM,
                 bidirectional: bool = False,
                 batch_norm: bool = True):

        super(BatchRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        # apply time-distributed batch normalization
        self.batch_norm = SequenceWise(
            nn.BatchNorm1d(input_size)) if batch_norm else None

        self.rnn = rnn_type(input_size=input_size,
                            hidden_size=hidden_size,
                            bidirectional=bidirectional,
                            bias=True)
        self.num_directions = 2 if bidirectional else 1

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x: torch.Tensor, output_lengths: torch.Tensor):

        if self.batch_norm is not None:
            x = self.batch_norm(x)

        x = nn.utils.rnn.pack_padded_sequence(x, output_lengths)
        x, h = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x)

        # sum forward and backward contexts if bidirectional
        if self.bidirectional:
            x = x.view(
                x.size(0), x.size(1), 2, -1
            ).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH)
        return x


class Lookahead(nn.Module):
    """
    Lookahead Convolution Layer for Unidirectional Recurrent Neural Networks
    from Wang et al 2016.
    """
    def __init__(self, n_features: int, context: int):
        """
        Parameters
        ----------
        n_features (int): feature dimension
        context (int):    context length in frames, corresponding to a lookahead
                          of (context - 1) frames
        """
        super(Lookahead, self).__init__()

        assert context > 0, 'Must provide nonzero context length'

        self.context = context
        self.n_features = n_features

        # pad to preserve sequence length in output
        self.pad = (0, self.context - 1)

        self.conv = nn.Conv1d(
            self.n_features,
            self.n_features,
            kernel_size=self.context,
            stride=1,
            groups=self.n_features,
            padding=0,
            bias=False
        )

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x (Tensor): shape (seq_len, n_batch, n_features)

        Returns
        -------
        out (Tensor): shape (seq_len, n_batch, n_features)
        """
        x = x.transpose(0, 1).transpose(1, 2)
        x = F.pad(x, pad=self.pad, value=0)
        x = self.conv(x)
        x = x.transpose(1, 2).transpose(0, 1).contiguous()
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'n_features=' + str(self.n_features) \
               + ', context=' + str(self.context) + ')'


class DeepSpeech(nn.Module):
    def __init__(self,
                 window_size: float = 0.02,
                 window_stride: float = 0.01,
                 normalize: bool = True):
        """
        Parameters
        ----------

        """

        super().__init__()

        # hard-code to match pre-trained implementation
        self.sample_rate = 16000
        self.labels = [
            '_', "'", 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
            'W', 'X', 'Y', 'Z', '|']
        self.sep_idx = len(self.labels) - 1
        self.blank_idx = 0
        self.hidden_size = 1024
        self.hidden_layers = 5
        self.lookahead_context = 0
        self.bidirectional: bool = True
        self.normalize = normalize
        num_classes = len(self.labels)

        # check sample rate
        if DataProperties.get("sample_rate") != self.sample_rate:
            raise ValueError(f"Incompatible data and model sample rates "
                             f"{DataProperties.get('sample_rate')}, "
                             f"{self.sample_rate}")

        # spectrogram processing - matches original Librosa implementation
        # (MSE ~1e-11 for 4s audio)
        self.spec = Spectrogram(
            n_fft=int(self.sample_rate * window_size),
            win_length=int(self.sample_rate * window_size),
            hop_length=int(self.sample_rate * window_stride),
            window_fn=torch.hamming_window,
            center=True,
            pad_mode='constant',
            power=1
        )

        # convolutional spectrogram encoder (acoustic model)
        self.conv = MaskConv(nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        ))

        # compute RNN input size using conv formula (W - F + 2P)/ S+1
        rnn_input_size = int(math.floor((self.sample_rate * window_size) / 2) + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 20 - 41) / 2 + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 10 - 21) / 2 + 1)
        rnn_input_size *= 32

        # RNN stack
        self.rnns = nn.Sequential(
            BatchRNN(
                input_size=rnn_input_size,
                hidden_size=self.hidden_size,
                rnn_type=nn.LSTM,
                bidirectional=self.bidirectional,
                batch_norm=False
            ),
            *(
                BatchRNN(
                    input_size=self.hidden_size,
                    hidden_size=self.hidden_size,
                    rnn_type=nn.LSTM,
                    bidirectional=self.bidirectional
                ) for x in range(self.hidden_layers - 1)
            )
        )

        # post-RNN lookahead (for unidirectional models)
        self.lookahead = nn.Sequential(
            Lookahead(self.hidden_size, context=self.lookahead_context),
            nn.Hardtanh(0, 20, inplace=True)
        ) if not self.bidirectional else None

        # final time-distributed linear layer for token prediction
        fully_connected = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size),
            nn.Linear(self.hidden_size, num_classes, bias=False)
        )
        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )
        self.inference_softmax = InferenceBatchSoftmax()

    def forward(self, x, lengths=None):
        """
        Parameters
        ----------
        x (Tensor):

        lengths (Tensor):
        """

        # ensure RNN blocks are in train mode to allow backpropagation for
        # attack optimization
        if not self.rnns.training:
            self.rnns.train()

        # require batch, channel dimensions
        assert x.ndim >= 2
        n_batch, *channel_dims, signal_len = x.shape

        if x.ndim == 2:
            x = x.unsqueeze(1)

        # convert to mono audio
        x = x.mean(dim=1, keepdim=True)

        # compute spectrogram
        x = self.spec(x)  # (n_batch, 1, n_freq, n_frames)
        x = torch.log1p(x)

        if self.normalize:
            mean = x.mean()
            std = x.std()
            x = x - mean
            x = x / std

        lengths = lengths or torch.full((n_batch,), x.shape[-1], dtype=torch.long)

        lengths = lengths.cpu().int()
        output_lengths = self.get_seq_lens(lengths)
        x, _ = self.conv(x, output_lengths)

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH

        for rnn in self.rnns:
            x = rnn(x, output_lengths)

        if not self.bidirectional:  # no need for lookahead layer in bidirectional
            x = self.lookahead(x)

        x = self.fc(x)
        x = x.transpose(0, 1)

        return x

    def get_seq_lens(self, input_length):
        """
        Given a 1D Tensor or Variable containing integer sequence lengths, return a 1D tensor or variable
        containing the size sequences that will be output by the network.
        :param input_length: 1D Tensor
        :return: 1D Tensor scaled by model
        """
        seq_len = input_length
        for m in self.conv.modules():
            if type(m) == nn.modules.conv.Conv2d:
                seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) // m.stride[1] + 1)
        return seq_len.int()
