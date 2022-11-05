import argbind
import sounddevice as sd
import numpy as np
import yaml
import torch
import os
from typing import Union

import sys
import warnings

sys.path.append('.')
warnings.filterwarnings('ignore', category=UserWarning)

from src.data.dataproperties import DataProperties
from src.attacks.online import Streamer, VoiceBoxStreamer
from src.constants import MODELS_DIR, CONDITIONING_FILENAME


def get_streams(input_name: str, output_name: str, block_size: int) -> tuple[sd.InputStream, sd.OutputStream]:
    """
    Gets Input and Output stream objects
    """
    try:
        input_name = int(input_name)
    except ValueError:
        pass
    try:
        output_name = int(output_name)
    except ValueError:
        pass
    return (
        sd.InputStream(device=input_name,
                       samplerate=DataProperties.get('sample_rate'),
                       channels=1,
                       blocksize=block_size),
        sd.OutputStream(device=output_name,
                        samplerate=DataProperties.get('sample_rate'),
                        channels=1,
                        blocksize=block_size)
    )


def get_model_streamer(device: str, conditioning_path: str) -> Streamer:
    # TODO: Make a good way to query an attack type. For now, I'm going to hard code this.
    model_dir = os.path.join(MODELS_DIR, 'voicebox')
    checkpoint_path = os.path.join(model_dir, 'voicebox_final.pt')
    config_path = os.path.join(model_dir, 'voicebox_final.yaml')

    with open(config_path) as f:
        config = yaml.safe_load(f)

    state_dict = torch.load(checkpoint_path, map_location=device)
    condition_tensor = torch.load(conditioning_path, map_location=device)
    model = VoiceBoxStreamer(
        **config
    )
    model.load_state_dict(state_dict)
    model.condition_vector = condition_tensor.reshape(1, 1, -1)

    streamer = Streamer(
        model=model,
        device=device,
        lookahead_frames=config['bottleneck_lookahead_frames'],
        recurrent=True
    )
    return streamer


def to_model(x: np.ndarray, device: str) -> torch.Tensor:
    return torch.Tensor(x).view(1, 1, -1).to(device)


def from_model(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().view(-1, 1).numpy()


@argbind.bind(without_prefix=True)
def main(
    input: str = None,
    output: str = '',
    device: str = 'cpu',
    num_frames: int = 4,
    pass_through: bool = False,
    conditioning_path: str = CONDITIONING_FILENAME
):
    f"""
    Uses a streaming implementation of an attack to perturb incoming audio

    :param input: Index or name of input audio interface. Defaults to current device
    :type input: str, optional
    :param output: Index of name output audio interface. Defaults to 0
    :type output: str, optional
    :param device: Device to processing attack. Should be either 'cpu' or 'cuda:X'
        Defaults to 'cpu'.
    :type device: str, optional
    :param pass_through: If True, the voicebox perturbation is not applied and the input will be
        identical to the output. This is for demo purposes. The input and output audio will
        remain at 16 kHz.
    :type pass_through: bool, optional
    :type device: str, optional
    :param num_frames: Number of overlapping model frames to process at one iteration.
        Defaults to 1
    :type num_frames: int
    :param conditioning_path: Path to conditioning tensor. Default: {CONDITIONING_FILENAME}
    :type conditioning_path: str
    """
    streamer = get_model_streamer(device, conditioning_path)
    input_stream, output_stream = get_streams(input, output, streamer.hop_length)
    if streamer.win_type in ['hann', 'triangular']:
        input_samples = (num_frames - 1) * streamer.hop_length + streamer.window_length
    else:
        input_samples = streamer.hop_length
    print("Ready to process audio")
    input_stream.start()
    output_stream.start()
    try:
        while True:
            frames, overflow = input_stream.read(input_samples)
            if pass_through:
                output_stream.write(frames)
                continue
            out = streamer.feed(to_model(frames, device))
            out = from_model(out)
            underflow = output_stream.write(out)
    except KeyboardInterrupt:
        print("Stopping")
        input_stream.stop()
        output_stream.stop()


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        main()
