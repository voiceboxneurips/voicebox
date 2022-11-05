import torch
import librosa
import soundfile as sf

from tqdm import tqdm
from src.attacks.offline.perturbation.voicebox import projection
from src.attacks.online import Streamer, VoiceBoxStreamer
from src.models import ResNetSE34V2, SpeakerVerificationModel
from src.constants import MODELS_DIR, TEST_DIR, PPG_PRETRAINED_PATH

import warnings
warnings.filterwarnings("ignore")

torch.set_num_threads(1)

device = 'cpu'

lookahead = 5

signal_length = 64_000
chunk_size = 640

test_audio = torch.Tensor(
    librosa.load(TEST_DIR / 'data' / 'test.wav', sr=16_000, mono=True)[0]
).unsqueeze(0).unsqueeze(0)

tests = [
    (512, 512, 512)
]
resnet_model = SpeakerVerificationModel(model=ResNetSE34V2())
condition_vector = resnet_model(test_audio)
for (bottleneck_hidden_size,
     bottleneck_feedforward_size,
     spec_encoder_hidden_size) in tests:
    print(
f"""
====================================
bottleneck_hidden_size: {bottleneck_hidden_size}
bottleneck_feedforward_size: {bottleneck_feedforward_size}
spec_encoder_hidden_size: {spec_encoder_hidden_size}
"""
    )

    streamer = Streamer(
        VoiceBoxStreamer(
            win_length=256,
            bottleneck_type='lstm',
            bottleneck_skip=True,
            bottleneck_depth=2,
            bottleneck_lookahead_frames=5,
            bottleneck_hidden_size=bottleneck_hidden_size,
            bottleneck_feedforward_size=bottleneck_feedforward_size,
        
            conditioning_dim=512,

            spec_encoder_mlp_depth=2,
            spec_encoder_hidden_size=spec_encoder_hidden_size,
            spec_encoder_lookahead_frames=0,
            ppg_encoder_path=PPG_PRETRAINED_PATH,
            
            ppg_encoder_depth=2,
            ppg_encoder_hidden_size=256,
            projection_norm='inf',
            control_eps=0.5,
            n_bands=128
        ),
        device,
        hop_length=128,
        window_length=256,
        win_type='hann',
        lookahead_frames=lookahead,
        recurrent=True
    )
    streamer.model.load_state_dict(torch.load(MODELS_DIR / 'voicebox' / 'voicebox_final.pt'))
    streamer.condition_vector = condition_vector

    output_chunks = []
    for i in tqdm(range(0, signal_length, chunk_size)):
        signal_chunk = test_audio[..., i:i+chunk_size]
        out = streamer.feed(signal_chunk)
        output_chunks.append(out)
    output_chunks.append(streamer.flush())
    output_audio = torch.cat(output_chunks, dim=-1)
    output_embedding = resnet_model(output_audio)

    print(
f"""
RTF: {streamer.real_time_factor}
Embedding Distance: {resnet_model.distance_fn(output_embedding, condition_vector)}
====================================
"""
        )
    sf.write(
        'output.wav',
        output_audio.numpy().squeeze(),
        16_000,
    )
