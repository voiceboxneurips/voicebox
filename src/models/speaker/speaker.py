import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.model import Model

################################################################################
# Extends Model class for speaker verification
################################################################################


class EmbeddingDistance(nn.Module):
    """
    Compute average pair-wise embedding distances over all segments of
    corresponding embeddings
    """

    def __init__(self, distance_fn: str = 'cosine'):
        super().__init__()

        self.distance_fn = distance_fn

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        Compute mean pairwise distance over all embedding pairs

        :param x1: embeddings of shape (n_batch, n_segments, embedding_dim)
        :param x2: embeddings of shape (n_batch, n_segments, embedding_dim)
        :return: mean pairwise distances of shape (n_batch,)
        """

        # match device
        assert x1.device == x2.device, "Input device mismatch"

        # reshape to (n_batch, n_segments, embedding_dim)
        try:

            assert x1.ndim >= 2
            assert x2.ndim >= 2

            n_batch, embedding_dim = x1.shape[0], x1.shape[-1]

            x1 = x1.reshape(n_batch, -1, embedding_dim)
            x2 = x2.reshape(n_batch, -1, embedding_dim)

            assert x1.shape == x2.shape

        except (AssertionError, RuntimeError):

            raise ValueError(f'Invalid input shapes {x1.shape}, '
                             f'{x2.shape}; embeddings should have shape '
                             f'(n_batch, n_segments, embedding_dim)')

        # compute distances over all segment pairs
        if self.distance_fn == 'l2':

            # normalize embeddings for L2 distance
            x1 = F.normalize(x1, p=2, dim=-1)
            x2 = F.normalize(x2, p=2, dim=-1)

            return torch.cdist(
                x1,
                x2,
                p=2.0
            ).reshape(-1, x1.shape[1] * x2.shape[1]).mean(-1)

        elif self.distance_fn == 'cosine':

            eps = 1e-8  # numerical stability

            dist = []

            for i in range(n_batch):

                x1_n = x1[i].norm(dim=1)[:, None]
                x2_n = x2[i].norm(dim=1)[:, None]

                x1_norm = x1[i] / torch.clamp(x1_n, min=eps)
                x2_norm = x2[i] / torch.clamp(x2_n, min=eps)

                sim = torch.mm(x1_norm, x2_norm.transpose(0, 1))

                dist_mtrx = 1 - sim

                dist.append(
                    dist_mtrx.reshape(-1, x1.shape[1] * x2.shape[1]).mean(-1)
                )

            return torch.cat(dist, dim=0)

        else:
            raise ValueError(f'Invalid embedding distance {self.distance_fn}')


class SpeakerVerificationModel(Model):
    """
    Perform speaker verification using a distance measured in the embedding
    space of a given model. If the distance between utterance embeddings exceeds
    a stored threshold, the utterances are assumed to originate from different
    speakers.
    """
    def __init__(self,
                 model: nn.Module,
                 n_segments: int = 1,
                 segment_select: str = 'lin',
                 distance_fn: str = 'cosine',
                 threshold: float = 0.0
                 ):
        """
        Wrap speaker verification model.

        :param model: a callable nn.Module object that produces speaker
                      embeddings. For inputs of shape (n_batch, signal_length)
                      or (n_batch, n_channels, signal_length), must produce
                      outputs of shape (n_batch, embedding_dim)
        :param n_segments: number of segments per utterance from which to
                           compute speaker embeddings. One embedding is produced
                           per segment, resulting in outputs of shape
                           (n_batch, n_segments, embedding_dim)
        :param segment_select: method for selecting utterance segments. Must be
                               `lin` (linearly-spaced) or `rand` (random)
        :param distance_fn:
        :param threshold: verification threshold in embedding space, according
                          to stored distance function
        """

        super().__init__()

        self.model = model
        self.n_segments = n_segments
        self.segment_select = segment_select
        self.threshold = threshold

        frame_len, hop_len = 400, 160  # 25ms frame / 10ms hop at 16kHz
        self.segment_frames = 400  # cap input segments at 400 frames
        self.segment_len = self.segment_frames * hop_len + frame_len - hop_len

        # check input segmentation method
        if segment_select not in ['lin', 'rand']:
            raise ValueError(f'Invalid segment selection method'
                             f' {segment_select}')
        self.segment_select = segment_select

        # prepare to compute pair-wise segment embedding distances
        self.distance_fn = EmbeddingDistance(distance_fn)

        # store distance function and threshold to allow prediction-matching
        self.threshold = threshold

        self.model.eval()

    def _pad_to_length(self, x: torch.Tensor):
        """
        Pad audio to stored segment length
        """
        if x.shape[-1] < self.segment_len:
            return nn.functional.pad(
                x,
                (0, self.segment_len - x.shape[-1])
            )
        else:
            return x

    def _extract_segments(self, x: torch.Tensor):
        """
        Given number of segments to extract and segment length, either space
        linearly or randomly. Should convert input of shape
        (n_batch, signal_length) to (n_batch * n_segments, segment_length) where
        segments from same utterance are consecutive along batch dimension
        """

        # if `n_segments` is nonzero, extract or trim audio to fixed-length
        # segments before computing embeddings
        if self.n_segments >= 1:

            # pad to allow a minimum of `n_segments` segments at set hop length
            min_hop_length = 400  # fold is very memory-hungry
            min_audio_length = self.segment_len + min_hop_length * (self.n_segments - 1)

            if x.shape[-1] < min_audio_length:
                x = nn.functional.pad(x, (0, min_audio_length - x.shape[-1]))

            # compute segment indices
            if self.segment_select == 'lin':  # linear spacing

                hop = (
                              x.shape[-1] - self.segment_len
                      ) // (
                              self.n_segments - 1
                      ) if self.n_segments > 1 else x.shape[-1]
                x = x.unfold(
                    -1, self.segment_len, hop
                )[:, :self.n_segments, :].contiguous()
                x = x.reshape(-1, self.segment_len)

            elif self.segment_select == 'rand':

                # slice at fine resolution, and randomly select segments
                x = x.unfold(
                    -1, self.segment_len, min_hop_length
                )

                x = x[:, torch.randperm(x.shape[1]), :]
                x = x[:, :self.n_segments, :].contiguous()
                x = x.reshape(-1, self.segment_len)

            else:
                raise ValueError(
                    f'Invalid segment selection method {self.segment_select}'
                )

        return x

    def forward(self, x: torch.Tensor):
        """
        Compute embeddings for input. If specified, divide inputs into segments
        and compute embeddings for each.
        """

        # reshape audio
        assert x.ndim >= 2  # require batch dimension
        n_batch, signal_len = x.shape[0], x.shape[-1]
        x = x.reshape(n_batch, signal_len)

        # pad to minimum length
        x = self._pad_to_length(x)

        # divide into segments
        x = self._extract_segments(x)  # (n_batch * n_segments, segment_length)

        # compute embeddings
        x = self.model(x)

        # group segments by corresponding utterance
        x = x.reshape(
            n_batch,
            1 if self.n_segments < 1 else self.n_segments,
            -1
        )  # (n_batch, n_segments, segment_length)

        return x  # return un-normalized embeddings by default

    def match_predict(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Determine whether target pairs are equivalent by checking if embedding
        distance falls under stored threshold
        """

        dist = self.distance_fn(y_pred, y_true)

        return dist <= self.threshold

    def load_weights(self, path: str):
        """
        Load weights from checkpoint file
        """

        # check if file exists
        if not path or not os.path.isfile(path):
            raise ValueError(f'Invalid path {path}')

        model_state = self.model.state_dict()
        loaded_state = torch.load(path)

        for name, param in loaded_state.items():

            origname = name

            if name not in model_state:
                print("{} is not in the model.".format(origname))
                continue

            if model_state[name].size() != loaded_state[origname].size():
                print(
                    "Wrong parameter length: {}, model: {}, loaded: {}".format(
                        origname,
                        model_state[name].size(),
                        loaded_state[origname].size()
                    )
                )
                continue

            model_state[name].copy_(param)

