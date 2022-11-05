import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import psutil
import pickle
import librosa as li

from torch.utils.data import TensorDataset

import time
import random
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from typing import Dict
from pathlib import Path
from tqdm import tqdm
import builtins

from typing import Iterable
from copy import deepcopy

from distutils.util import strtobool

from src.data import *
from src.constants import *
from src.models import *
from src.simulation import *
from src.preprocess import *
from src.attacks.offline import *
from src.loss import *
from src.pipelines import *
from src.utils import *

################################################################################
# Train VoiceBox attack
################################################################################

BATCH_SIZE = 20     # training batch size
EPOCHS = 10         # training epochs
TARGET_PCTL = 25    # de-identification strength; in [1,5,10,15,20,25,50,90,100]
N_EMBEDDINGS_TRAIN = 15
TARGETED = False
TARGETS_TRAIN = 'centroid'  # 'random', 'same', 'single', 'median'
TARGETS_TEST = 'centroid'  # 'random', 'same', 'single', 'median'

# distributions of inter- ('targeted') and intra- ('untargeted') speaker
# distances in each pre-trained model's embedding spaces, as measured between
# individual utterances and their speaker centroid ('single-centroid') or
# between all pairs of individual utterances ('single-single') over the
# LibriSpeech test-clean dataset. This allows specification of attack strength
# during the training process
percentiles = {
    'resnet': {
        'targeted': {
            'single-centroid': {1:.495, 5:.572, 10:.617, 15:.648, 20:.673, 25:.695, 50:.773, 90:.892, 100:1.127},
            'single-single': {1:.560, 5:.630, 10:.672, 15:.700, 20:.722, 25:.742, 50:.813, 90:.924, 100:1.194}
        },
        'untargeted': {
            'single-centroid': {1:.099, 5:.117, 10:.126, 15:.133, 20:.139, 25:.145, 50:.170, 90:.253, 100:.587},
            'single-single': {1:.181, 5:.215, 10:.235, 15:.249, 20:.262, 25:.272, 50:.323, 90:.464, 100:.817}
        },
    },
    'yvector': {
        'targeted': {
            'single-centroid': {1:.665, 5:.757, 10:.801, 15:.830, 20:.851, 25:.868, 50:.936, 90:1.056, 100:1.312},
            'single-single': {1:.695, 5:.779, 10:.821, 15:.847, 20:.868, 25:.885, 50:.952, 90:1.072, 100:1.428}
        },
        'untargeted': {
            'single-single': {1:.218, 5:.268, 10:.301, 15:.325, 20:.345, 25:.365, 50:.455, 90:.684, 100:1.156},
            'single-centroid': {1:.114, 5:.143, 10:.159, 15:.170, 20:.180, 25:.190, 50:.242, 90:.413, 100:.874}
        }
    },
}


def set_random_seed(seed: int = 123):
    """Set random seed to allow for reproducibility"""
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.backends.cudnn.is_available():
        # torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True


def param_count(m: nn.Module, trainable: bool = False):
    """Count the number of trainable parameters (weights) in a model"""
    if trainable:
        return builtins.sum(
            [p.shape.numel() for p in m.parameters() if p.requires_grad])
    else:
        return builtins.sum([p.shape.numel() for p in m.parameters()])


def main():

    set_random_seed(0)

    model = SpeakerVerificationModel(
        model=ResNetSE34V2(nOut=512, encoder_type='ASP'),
        n_segments=1,
        segment_select='lin',
        distance_fn='cosine',
        threshold=percentiles['resnet']['targeted']['single-centroid' if
        TARGETS_TRAIN == 'centroid' else 'single-single'][TARGET_PCTL]
    )
    model.load_weights(MODELS_DIR / 'speaker' / 'resnetse34v2' / 'resnetse34v2.pt')

    # instantiate training pipeline
    pipeline = Pipeline(
        simulation=None,
        preprocessor=Preprocessor(Normalize(method='peak')),
        model=model,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    attacks = {}

    # log training progress
    writer = Writer(
        root_dir=RUNS_DIR,
        name='train-attacks',
        use_timestamp=True,
        log_iter=300,
        use_tb=True
    )

    # adversarial training loss
    adv_loss = SpeakerEmbeddingLoss(
        targeted=TARGETED,
        confidence=0.1,
        threshold=pipeline.model.threshold
    )

    # auxiliary loss
    aux_loss = SumLoss().add_loss_function(
        DemucsMRSTFTLoss(), 1.0
    ).add_loss_function(L1Loss(), 1.0).to('cuda')

    # speech features loss actually seems to do better...
    # aux_loss = SumLoss().add_loss_function(SpeechFeatureLoss(), 1e-6).to('cuda')

    attacks['voicebox'] = VoiceBoxAttack(
        pipeline=pipeline,
        adv_loss=adv_loss,
        aux_loss=aux_loss,
        lr=1e-4,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        voicebox_kwargs={
            'win_length': 256,
            'ppg_encoder_hidden_size': 256,
            'use_phoneme_encoder': True,
            'use_pitch_encoder': True,
            'use_loudness_encoder': True,
            'spec_encoder_lookahead_frames': 0,
            'spec_encoder_type': 'mel',
            'spec_encoder_mlp_depth': 2,
            'bottleneck_lookahead_frames': 5,
            'ppg_encoder_path': PPG_PRETRAINED_PATH,
            'n_bands': 128,
            'spec_encoder_hidden_size': 512,
            'bottleneck_skip': True,
            'bottleneck_hidden_size': 512,
            'bottleneck_feedforward_size': 512,
            'bottleneck_type': 'lstm',
            'bottleneck_depth': 2,
            'control_eps': 0.5,
            'projection_norm': float('inf'),
            'conditioning_dim': 512
        },
        writer=writer,
        checkpoint_name='voicebox-attack'
    )

    attacks['universal'] = AdvPulseAttack(
        pipeline=pipeline,
        adv_loss=adv_loss,
        pgd_norm=float('inf'),
        pgd_variant=None,
        scale_grad=None,
        eps=0.08,
        length=2.0,
        align='random',  # 'start',
        lr=1e-4,
        normalize=True,
        loop=True,
        aux_loss=aux_loss,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        writer=writer,
        checkpoint_name='universal-attack'
    )

    if torch.cuda.is_available():

        # prepare for multi-GPU training
        device_ids = get_cuda_device_ids()

        # wrap pipeline for multi-GPU training
        pipeline = wrap_pipeline_multi_gpu(pipeline, device_ids)

    # load training and validation datasets. Features will be computed and
    # cached to disk, which may take some time
    data_train = LibriSpeechDataset(
        split='train-clean-100', features=['pitch', 'periodicity', 'loudness'])
    data_test = LibriSpeechDataset(
        split='test-clean', features=['pitch', 'periodicity', 'loudness'])

    # reassign targets if necessary
    compiled_train, compiled_test = create_embedding_dataset(
        pipeline=pipeline,
        select_train=TARGETS_TRAIN,
        select_test=TARGETS_TEST,
        data_train=data_train,
        data_test=data_test,
        targeted=TARGETED,
        target_class=None,
        num_embeddings_train=N_EMBEDDINGS_TRAIN,
        batch_size=20
    )

    # extract embedding datasets
    data_train = compiled_train['dataset']
    data_test = compiled_test['dataset']

    # log memory use prior to training
    writer.log_info(f'Training data ready; memory use: '
                    f'{psutil.virtual_memory().percent :0.3f}%')
    writer.log_cuda_memory()

    for attack_name, attack in attacks.items():

        writer.log_info(f'Preparing {attack_name}...')

        if torch.cuda.is_available():

            attack.perturbation.to('cuda')
            attack.pipeline.to('cuda')

            # wrap attack for multi-GPU training
            attack = wrap_attack_multi_gpu(attack, device_ids)

        # evaluate performance
        with torch.no_grad():
            x_example = next(iter(data_train))['x'].to(pipeline.device)
            st = time.time()
            outs = attack.perturbation(x_example)
            dur = time.time() - st

        writer.log_info(
            f'Processing time per input (device: '
            f'{pipeline.device}): {dur/x_example.shape[0] :0.4f} (s)'
        )
        writer.log_info(f'Trainable parameters: '
                        f'{param_count(attack.perturbation, trainable=True)}')
        writer.log_info(f'Total parameters: {param_count(attack.perturbation, trainable=False)}')

        # train
        writer.log_info('Training attack...')
        attack.train(data_train=data_train, data_val=data_test)

        # evaluate
        writer.log_info(f'Evaluating attack...')
        x_adv, success, detection = attack.evaluate(
            dataset=data_test
        )

        # log results summary: success rate in achieving target threshold
        writer.log_info(
            f'Success rate in meeting embedding distance threshold {pipeline.model.threshold}'
            f' ({TARGET_PCTL}%): '
            f'{success.flatten().mean().item()}'
        )


if __name__ == "__main__":
    main()
