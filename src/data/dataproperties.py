import torch

################################################################################
# Allow project-wide access to persistent data properties
################################################################################


class DataProperties(object):
    """
    Allow shared access to data properties (e.g. sample rate) across all audio
    processing modules. Each dataset registers its properties with the
    DataProperties class upon initialization, eliminating the need to repeatedly
    pass properties as parameters
    """

    # Default data properties: 1-second 16kHz audio scaled to [-1, 1]
    properties = {
        "sample_rate": 16000,
        "scale": 1.0,
        "signal_length": 16000
    }

    @classmethod
    def register_properties(cls, **kwargs):
        """
        Register data properties by name
        """
        cls.properties = kwargs

    @classmethod
    def get(cls, *args):
        """
        Access one or more data properties by name
        """
        if len(args) > 1:
            return tuple(cls.properties[a] for a in args)
        else:
            return cls.properties[args[0]]

    @classmethod
    def format_input(cls, x: torch.Tensor):
        """
        Ensure input is correctly formatted (batch/channels/samples). If input
        cannot be reshaped to required dimensions, raise error
        """

        try:
            signal_length = cls.properties["signal_length"]
        except KeyError:
            raise ValueError(f"Data property `signal_length` must be defined to"
                             f" format inputs")

        if x.ndim <= 1:
            n_batch = 1

        else:
            n_batch = x.shape[0]

        try:
            x = x.reshape(n_batch, 1, signal_length)
        except RuntimeError:
            raise ValueError(f"Invalid input dimensions {list(x.shape)}")

        return x
