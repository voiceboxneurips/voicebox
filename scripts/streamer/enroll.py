"""
Pipeline for enrolling:
1. Provide Recording
2. Convert to 16 kHz
3. Divide into recordings
4. Get embeddings for each recording
5. Find centroid
6. Save conditioning as some value.
"""
import os
import argbind
import sounddevice as sd
import soundfile
import torch
import numpy as np

import sys

sys.path.append('.')

from src.constants import CONDITIONING_FILENAME, CONDITIONING_FOLDER
from src.data import DataProperties
from src.models import ResNetSE34V2


MIN_WINDOWS = 10
WINDOW_SIZE = 64_000
BLOCK_SIZE = 256

RECORDING_TEXT = """
This script will record you speaking, and will create an embedding
to be used for conditioning Voicebox. This will overwrite any previous
embeddings. We recommend at least 10 seconds of non-stop voice recording.
Press enter to begin recording. To stop recording, press ctrl-C.
"""


def get_streams(input_name: str, block_size: int) -> sd.InputStream:
    """
    Gets Input stream object
    """
    try:
        input_name = int(input_name)
    except ValueError:
        pass
    return (
        sd.InputStream(device=input_name,
                       samplerate=DataProperties.get('sample_rate'),
                       channels=1,
                       blocksize=block_size)
    )


def record_from_user(input_name: str) -> torch.Tensor:
    input_stream = get_streams(input_name, BLOCK_SIZE)
    input(RECORDING_TEXT)
    input_stream.start()
    all_frames = []
    try:
        print("Recording...")
        while True:
            frames, _ = input_stream.read(BLOCK_SIZE)
            all_frames.append(frames)
    except KeyboardInterrupt:
        print("Stopped Recording.")
        pass
    all_frames = torch.Tensor(np.array(all_frames))
    recording = all_frames.reshape(-1)
    return recording


def get_embedding(recording) -> torch.Tensor:
    model = ResNetSE34V2(nOut=512, encoder_type='ASP')
    recording = recording.view(1, -1)
    embedding = model(recording)
    return embedding


def save(embedding, audio) -> None:
    os.makedirs(CONDITIONING_FOLDER, exist_ok=True)
    torch.save(embedding, CONDITIONING_FILENAME)
    soundfile.write(
        CONDITIONING_FOLDER / 'conditioning_audio.wav',
        audio.detach().cpu(),
        DataProperties.get('sample_rate')
    )


@argbind.bind(positional=True, without_prefix=True)
def main(input: str = None):
    """
    Creating a conditioning vector for VoiceBox from your voice

    :param input: Index or name of input audio interface. Defaults to current device
    :type input: str, optional
    """
    recording = record_from_user(input)
    embedding = get_embedding(recording)
    save(embedding, recording)


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        main()
