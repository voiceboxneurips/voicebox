import torch
import torch.nn as nn
import torch.nn.functional as F

from src.loss.auxiliary import AuxiliaryLoss
from src.models.speech import SpeechRecognitionModel, Wav2Vec2


################################################################################
# ASR feature-matching loss
################################################################################


class SpeechFeatureLoss(AuxiliaryLoss):
    """
    Compute distance at encoded feature representations or token emission
    probabilities produced by a pretrained ASR model. For speech audio, these
    representations should capture some notion of phonetic similarity. Adapted
    from https://bit.ly/3z6EGyR.
    """
    def __init__(self,
                 reduction: str = 'none',
                 model: SpeechRecognitionModel = SpeechRecognitionModel(
                     Wav2Vec2()
                 ),
                 use_tokens: bool = False
                 ):
        super().__init__(reduction)

        self.ref_feats = None
        self.ref_tokens = None

        self.model = model

        # disable gradient computation for ASR model parameters
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.use_tokens = use_tokens

    def _compute_loss(self, x: torch.Tensor, x_ref: torch.Tensor = None):
        """
        Compute distance at encoded feature representations or token emission
        probabilities produced by a pretrained ASR model.

        :param x: input, shape (n_batch, n_channels, signal_length)
        :param x_ref: reference, shape (n_batch, n_channels, signal_length) or
                      (1, n_channels, signal_length)
        :return: unreduced batch loss, shape (n_batch,)
        """

        # require batch dimension
        assert x.ndim >= 2
        n_batch = x.shape[0]

        # prepare to store unreduced batch loss
        loss = torch.zeros(x.shape[0]).to(x.device)

        # compare token emission probabilities
        if self.use_tokens:

            x_tokens = self.model(x)

            if x_ref is not None:
                x_ref_tokens = self.model(x_ref)
            else:
                x_ref_tokens = self.ref_tokens

            # check compatibility of input and reference emissions
            assert self._check_broadcastable(
                x_tokens, x_ref_tokens
            ), f"Cannot broadcast inputs of shape {x_tokens.shape} " \
               f"with reference of shape {x_ref_tokens.shape}"

            loss += (x_tokens - x_ref_tokens).reshape(
                n_batch,
                -1
            ).norm(p=1, dim=-1)

        # compare deep features
        else:

            x_feats = self.model.extract_features(x)

            if x_ref is not None:
                x_ref_feats = self.model.extract_features(x_ref)
            else:
                x_ref_feats = self.ref_feats

            for i in range(len(x_feats)):

                # check compatibility of input and reference features
                assert self._check_broadcastable(
                    x_feats[i], x_ref_feats[i]
                ), f"Cannot broadcast inputs of shape {x_feats[i].shape} " \
                   f"with reference of shape {x_ref_feats[i].shape}"

                loss += (x_feats[i] - x_ref_feats[i]).reshape(
                    n_batch,
                    -1
                ).norm(p=1, dim=-1)

        return loss

    def set_reference(self, x_ref: torch.Tensor):
        """
        Compute and store deep features and token emission probabilities with
        pretrained ASR model.
        """

        # store deep features
        self.ref_feats = [
            r.detach() for r in self.model.extract_features(x_ref)
        ]

        # store token emission probabilities
        self.ref_tokens = self.model(x_ref).detach()
