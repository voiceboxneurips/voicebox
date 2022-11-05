import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union

################################################################################
# Apply projection to regularize controls based on past context
################################################################################


class CausalControlProjection(nn.Module):
    """
    Constrain each frame's perturbation 'budget' (projection bound) based on
    perturbation magnitudes in previous frames. This may help to encourage
    sparser high-magnitude perturbations.

    (n_batch, n_frames, n_controls) --> (n_batch, n_frames, n_controls)
    """
    def __init__(self,
                 eps: float,
                 n_controls: int,
                 unity: float,
                 projection_norm: Union[str, float, int] = 2,
                 method: str = 'exponential',
                 decay: float = 2,
                 context: int = 10, *args, **kwargs):
        """
        Parameters
        ----------
        eps (float):      perturbation bound

        n_controls (int): dimension of control vector for each frame

        unity (float):    'neutral' value for controls; will be used to center
                          during projection

        projection_norm:  norm for projection (2 or infinity)

        method (str):     strategy for regularizing controls based on past
                          context:

                            'none': apply projection with bound `eps` to all
                            frames independently

                            'exponential': controls bound at any frame is `eps`
                            minus an exponentially-decaying weighted average of
                            the control magnitudes of frames within the
                            preceding context window

                            'max': controls bound at any frame is `eps` minus
                            the maximum perturbation magnitude over the
                            preceding context window

        decay (float):    decay rate of exponential moving average in frames,
                          i.e. after `decay` steps a frame's contribution to the
                          average is scaled by a factor of 1/e

        context (int):    number of frames considered; frames beyond the context
                          window are truncated/removed from consideration
        """
        super().__init__()

        if eps is None:
            projection_norm = None

        self.eps = eps
        self.n_controls = n_controls
        self.unity = unity

        assert projection_norm in [
            "none", None, 2, 2.0, "2", float("inf"), "inf"
        ]
        self.projection_norm = projection_norm

        assert method in ['none', None, 'exp', 'exponential', 'max', 'maximum'], \
            f'Invalid causal regularization method {method}'

        self.method = method
        self.decay = decay
        self.context = context

        # compute exponential decay factor alpha
        a_ = math.exp(-1 / self.decay)
        precision = 3
        self.alpha = 1.0

        while self.alpha >= 1.0:
            self.alpha = round(a_, precision)
            precision += 1

        # compute "kernel" for exponentially-weighted average over context
        # window, and reshape to broadcast with "unfolded" inputs of shape
        # (n_batch, n_frames, self.context, 1)
        self.exp_kernel = torch.as_tensor(
            [self.alpha**i for i in range(1, self.context + 1)][::-1]
        ).reshape(1, 1, -1, 1).float()

    def _project(self, x: torch.Tensor, eps: Union[float, torch.Tensor]):
        """
        Apply frame-wise projection with given bound.

        Parameters
        ----------
        x (Tensor): shape (n_batch, n_frames, n_controls)

        Returns
        -------
        projected (Tensor): shape (n_batch, n_frames, n_controls)
        """

        if isinstance(eps, float):
            eps = torch.tensor(eps, device=x.device)

        # L2 projection
        if self.projection_norm in [2, '2', 2.0]:

            norm = torch.norm(
                x, p=2, dim=-1, keepdim=True) + 1e-20
            factor = torch.min(
                torch.tensor(1., device=x.device),
                eps / norm
            )
            x = x * factor

        # L-infinity projection
        elif self.projection_norm in [float('inf'), 'inf']:

            x = torch.clamp(
                x,
                min=-eps.abs(),
                max=eps.abs()
            )

        return x

    def forward(self, x: torch.Tensor):
        """
        Regularize control signal via projection.

        Parameters
        ----------
        x (Tensor): shape (n_batch, n_frames, n_controls)

        Returns
        -------
        projected (Tensor): shape (n_batch, n_frames, n_controls)
        """

        # optionally, perform no projection
        if self.projection_norm in ["none", None]:
            return x

        n_batch, n_frames, n_controls = x.shape
        assert n_controls == self.n_controls

        # center at unity
        unity = torch.full(x.shape, self.unity, device=x.device)
        x = x - unity

        # project controls for each frame independently
        if self.method in ['none', None]:
            budget = self.eps

        # project controls based on past context
        else:

            # compute control magnitudes for each frame
            magnitudes = x.abs().norm(
                dim=-1,
                keepdim=True,
                p=2 if self.projection_norm in [2, 2.0, "2"] else float("inf")
            )

            # apply left-padding to generate one context window per input frame
            padded = F.pad(magnitudes, (0, 0, self.context - 1, 0))

            # get all (overlapping) context "windows" with stride 1
            windows = padded.unfold(
                dimension=1,
                size=self.context,
                step=1
            ).permute(0, 1, 3, 2)  # (n_batch, n_frames, self.context, 1)

            # determine frame-wise projection bounds from exponentially-decaying
            # weighted average of control amplitudes within context window
            if self.method in ['exp', 'exponential']:

                # compute weighted averages with kernel / sum
                averages = (windows * self.exp_kernel).sum(dim=2)  # (n_batch, n_frames, 1)

                # compute projection bound for each frame
                budget = (self.eps - averages).clamp(min=0)  # (n_batch, n_frames, 1)

            # determine frame-wise projection bounds from maximum control amplitudes
            # within context window
            elif self.method in ['max', 'maximum']:

                # take maximum over each window
                maxima = torch.max(windows, dim=2)[0]  # (n_batch, n_frames, 1)

                # compute projection bound for each frame
                budget = (self.eps - maxima).clamp(min=0)  # (n_batch, n_frames, 1)

            else:
                raise ValueError(f'Invalid regularization method {self.method}')

        # apply frame-wise projection
        x = self._project(x, budget)

        # undo centering
        x = x + unity

        return x
