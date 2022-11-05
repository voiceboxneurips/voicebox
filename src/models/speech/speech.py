import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from typing import Union, List, Tuple

from src.models.model import Model

################################################################################
# Extends Model class for speech recognition, with optional decoding
################################################################################


class Decoder(object):
    """
    Code adapted from DeepSpeech PyTorch (https://tinyurl.com/2p89d35e). Base
    class for decoder objects, which convert emitted frame-by-frame token
    probabilities into a string transcription.
    """
    def __init__(self,
                 labels: Union[List[str], Tuple[str]],
                 sep_idx: int = None,
                 blank_idx: int = 0):
        """
        Parameters
        ----------
        labels (list):   character corresponding to each token index

        sep_idx (int):   index corresponding to space / separating character

        blank_idx (int): index corresponding to blank '_' character
        """
        self.labels = labels
        self.blank_idx = blank_idx
        self.int_to_char = dict([(i, c) for (i, c) in enumerate(labels)])

        if sep_idx is None:

            # use out-of-bounds index for separating character
            sep_idx = len(labels)
            if ' ' in labels:
                sep_idx = labels.index(' ')
            elif '|' in labels:
                sep_idx = labels.index('|')
            self.sep_idx = sep_idx

        else:
            self.sep_idx = sep_idx

    def get_labels(self):
        return self.labels

    def get_sep_idx(self):
        return self.sep_idx

    def get_blank_idx(self):
        return self.blank_idx

    def __call__(self, emission: torch.Tensor, sizes=None):
        return self.decode(emission, sizes)

    def decode(self, emission: torch.Tensor, sizes=None):
        """
        Decode emitted token probabilities to obtain a string transcription.

        Parameters
        ----------
        emission (Tensor): shape (n_batch, n_frames, n_tokens)

        sizes (Tensor):    length in frames of each emission in batch
        """
        raise NotImplementedError


class GreedyCTCDecoder(Decoder):
    """
    A simple decoder module to map token probability sequences to transcripts.
    Decodes 'greedily' by selecting maximum-probability token at each time step.
    Code adapted from DeepSpeech PyTorch (https://tinyurl.com/2p89d35e).
    """
    def __init__(self,
                 labels: Union[List[str], Tuple[str]],
                 sep_idx: int = None,
                 blank_idx: int = 0):
        super().__init__(labels, sep_idx, blank_idx)

    def convert_to_strings(self,
                           sequences,
                           sizes=None,
                           remove_repetitions=False,
                           return_offsets=False):
        """
        Given a list of sequences holding token numbers, return the
        corresponding strings. Optionally, collapse repeated token subsequences
        and return final length of each processed sequence.

        Parameters
        ----------

        sequences (Tensor): shape (n_batch, n_frames); holds argmax token index
                            for each frame

        sizes

        remove_repetitions

        return_offsets

        Returns
        -------

        """

        strings = []
        offsets = [] if return_offsets else None

        for i, sequence in enumerate(sequences):

            seq_len = sizes[i] if sizes is not None else len(sequence)
            string, string_offsets = self.process_string(sequence, seq_len, remove_repetitions)
            strings.append(string)
            if return_offsets:
                offsets.append(string_offsets)

        if return_offsets:
            return strings, offsets
        else:
            return strings

    def process_string(self,
                       sequence,
                       size,
                       remove_repetitions=False):
        string = ''
        offsets = []

        for i in range(size):
            char = self.int_to_char[sequence[i].item()]

            if char != self.int_to_char[self.blank_idx]:

                # skip repeated characters if specified
                if remove_repetitions and i != 0 and \
                        char == self.int_to_char[sequence[i - 1].item()]:
                    pass
                elif char == self.labels[self.sep_idx]:
                    string += self.labels[self.sep_idx]
                    offsets.append(i)
                else:
                    string = string + char
                    offsets.append(i)

        return string, torch.tensor(offsets, dtype=torch.int)

    def decode(self, emission, sizes=None):
        """
        Returns the argmax decoding given the emitted token probabilities.
        According to connectionist temporal classification (CTC), removes
        repeated elements in the decoded token sequence, as well as blanks.

        Parameters
        ----------
        emission (Tensor): shape (n_batch, n_frames, n_tokens)

        sizes (Tensor):    length in frames of each emission in batch

        Returns
        -------
        transcription (list[str]): string transcription for each item in batch

        offsets (???     frame index per character predicted

        """

        if emission.ndim == 2:  # require shape (n_batch, n_frames, n_tokens)
            emission = emission.unsqueeze(0)

        # compute max-probability label at each sequence index
        max_probs = torch.argmax(emission, dim=-1)  # (n_batch, sequence_len)

        strings, offsets = self.convert_to_strings(max_probs,
                                                   sizes,
                                                   remove_repetitions=True,
                                                   return_offsets=True)
        return strings, offsets


class SpeechRecognitionModel(Model):

    def __init__(self,
                 model: nn.Module,
                 decoder: Decoder = None
                 ):

        super().__init__()

        self.model = model
        self.model.eval()

        # ensure that list of viable tokens can be retrieved from wrapped model
        labels_method = getattr(self.model, "get_labels", None)
        labels_attr = getattr(self.model, "labels", None)
        if callable(labels_method):
            self._get_labels_fn = lambda: self.model.get_labels()
        elif labels_attr is not None:
            self._get_labels_fn = lambda: self.model.labels
        else:
            raise ValueError(f'Wrapped model must have method `.get_labels()`'
                             f' or attribute `.labels`')

        # ensure that blank and separator tokens can be retrieved from wrapped
        # model
        sep_method = getattr(self.model, "get_sep_idx", None)
        sep_attr = getattr(self.model, "sep_idx", None)
        if callable(sep_method):
            self._get_sep_fn = lambda: self.model.get_sep_idx()
        elif sep_attr is not None:
            self._get_sep_fn = lambda: self.model.sep_idx
        else:
            raise ValueError(f'Wrapped model must have method `.get_sep_idx()`'
                             f' or attribute `.sep_idx`')

        blank_method = getattr(self.model, "get_blank_idx", None)
        blank_attr = getattr(self.model, "blank_idx", None)
        if callable(blank_method):
            self._get_blank_fn = lambda: self.model.get_blank_idx()
        elif blank_attr is not None:
            self._get_blank_fn = lambda: self.model.blank_idx
        else:
            raise ValueError(f'Wrapped model must have method '
                             f'`.get_blank_idx()` or attribute `.blank_idx`')

        # initialize decoder
        if decoder is None:
            decoder = GreedyCTCDecoder(
                labels=self.get_labels(),
                blank_idx=self.get_blank_idx(),
                sep_idx=self.get_sep_idx()
            )
        self.decoder = decoder

        # translate characters to token indices
        self.char_to_idx = {l: i for i, l in enumerate(decoder.labels)}

    def get_labels(self):
        """Retrieve a list of valid tokens"""
        return self._get_labels_fn()

    def get_blank_idx(self):
        """Return index of blank token"""
        return self._get_blank_fn()

    def get_sep_idx(self):
        """Return index of separator token"""
        return self._get_sep_fn()

    def forward(self, x: torch.Tensor):
        return self.model.forward(x)

    def transcribe(self, x: torch.Tensor, return_alignment: bool = False):

        if return_alignment:
            return self.decoder(self.model(x))
        else:
            return self.decoder(self.model(x))[0]

    def load_weights(self, path: str):
        """
        Load weights from checkpoint file
        """

        # check if file exists
        if not path or not os.path.isfile(path):
            return

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

    def extract_features(
            self,
            x: torch.Tensor
    ) -> List[torch.Tensor]:

        """
        Extract deep features.

        :param x: input
        :return: a list of tensors holding intermediate activations / features
        """

        try:
            return self.model.extract_features(x)
        except AttributeError:
            return []

    def _str_to_tensor(self, seq: str):
        token_indices = [self.char_to_idx[c] for c in seq]
        return torch.as_tensor(token_indices, dtype=torch.long)

    def match_predict(self,
                      y_pred: Union[List[str], torch.Tensor],
                      y_true: Union[List[str], torch.Tensor]):
        """
        Determine whether (batched) target pairs are equivalent.
        """

        n_batch = len(y_pred)

        y_true_lengths = None

        # convert ground-truth transcriptions to tensor form
        if isinstance(y_true, list):
            y_true = [self._str_to_tensor(t) for t in y_true]
            y_true_lengths = [t.shape[-1] for t in y_true]
            y_true = pad_sequence(
                y_true,
                batch_first=True
            )  # (n_batch, max_seq_len)

        if y_true_lengths is None:
            y_true_lengths = [y_true.shape[-1]] * n_batch

        # convert predicted transcriptions to tensor form
        if isinstance(y_pred, list):
            y_pred = [self._str_to_tensor(t) for t in y_pred]
            y_pred = pad_sequence(
                y_pred,
                batch_first=True
            )  # (n_batch, max_seq_len)

        length_diff = max(0, y_true.shape[-1] - y_pred.shape[-1])
        if length_diff:
            y_pred = F.pad(y_pred, (0, length_diff))

        matches = []
        for i in range(n_batch):
            matches.append(
                torch.all(
                    y_pred[i, ..., :y_true_lengths[i]] == y_true[i, ..., :y_true_lengths[i]]
                )
            )

        return torch.as_tensor(matches)



        """
        # masked comparison
        use which one as dimension to select --- true or pred?

        pred lengths may be unnecessary! just select to true length
        """


