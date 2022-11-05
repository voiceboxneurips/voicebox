import torch

from src.simulation.component import Component

################################################################################
# Normalize audio
################################################################################


class Normalize(Component):

    def __init__(self,
                 method: str = 'peak',
                 target_dbfs: float = -30.0,
                 increase_only: bool = False,
                 decrease_only: bool = False
                 ):
        """
        Normalize incoming audio.

        Parameters
        ----------

        Returns
        -------

        """
        super().__init__()

        assert method in [None, 'none', 'peak', 'dbfs'], \
            f"Invalid normalization method {method}"
        self.method = method

        # parameters for dBFS normalization
        assert not (increase_only and decrease_only), \
            f"Cannot set both `increase_only` and `decrease_only`"

        self.target_dbfs = target_dbfs
        self.increase_only = increase_only
        self.decrease_only = decrease_only

    def forward(self, x: torch.Tensor):

        if self.method is None:
            return x
        elif self.method == 'peak':
            return (self.scale / torch.max(
                torch.abs(x) + 1e-8, dim=-1, keepdim=True)[0]) * x * 0.95
        elif self.method == 'dbfs':

            # compute volume in dBFS
            rms = torch.sqrt(torch.mean(x ** 2))
            dbfs = 20 * torch.log10(rms)

            # determine whether to normalize
            dbfs_change = self.target_dbfs - dbfs
            if dbfs_change < 0 and self.increase_only or \
                    dbfs_change > 0 and self.decrease_only:
                return x

            normalized = x * (10 ** (dbfs_change / 20))

            # clip to valid range
            return torch.clamp(normalized, min=-self.scale, max=self.scale)

        else:
            raise ValueError(f'Invalid normalization: {self.method}')
