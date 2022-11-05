import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import psutil
import pickle

import random
import argparse

import librosa as li
from sklearn.utils import shuffle
from sklearn.neighbors import NearestNeighbors

from pesq import pesq, NoUtterancesError

from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import numpy as np
from pathlib import Path
from tqdm import tqdm
import builtins
import math
import jiwer
from jiwer import wer, cer

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
# Evaluate attacks on speaker recognition systems
################################################################################

EVAL_DATASET = "voxceleb"  # "librispeech"
LOOKAHEAD = 5
VOICEBOX_PATH = VOICEBOX_PRETRAINED_PATH
UNIVERSAL_PATH = UNIVERSAL_PRETRAINED_PATH
BATCH_SIZE = 20     # evaluation batch size
N_QUERY = 15        # number of query utterances per speaker
N_CONDITION = 10    # number of conditioning utterances per speaker
N_ENROLL = 20       # number of enrolled utterances per speaker
ADV_ENROLL = False  # evaluate under assumption adversarial audio is enrolled
TARGETS_TRAIN = 'centroid'  # 'random', 'same', 'single', 'median'
TARGETS_TEST = 'centroid'  # 'random', 'same', 'single', 'median'
TRANSFER = True  # evaluate attacks on unseen model
DENOISER = False  # evaluate with unseen denoiser defense applied to queries
SIMULATION = False  # apply noisy channel simulation to all queries in evaluation
COMPUTE_OBJECTIVE_METRICS = True  # PESQ, STOI


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


def pad_sequence(sequences: list):

    max_len = max([s.shape[-1] for s in sequences])

    padded = torch.zeros(
        (len(sequences), 1, max_len),
        dtype=sequences[0].dtype,
        device=sequences[0].device)

    for i, s in enumerate(sequences):
        padded[i, :, :s.shape[-1]] = s

    return padded


@torch.no_grad()
def compute_embeddings_batch(audio: list,
                             p: Pipeline,
                             defense: nn.Module = nn.Identity()):
    """Compute batched speaker embeddings"""

    assert isinstance(p.model, SpeakerVerificationModel)
    emb = [p(defense(audio[i].to(p.device))).to('cpu') for i in range(len(audio))]
    emb = torch.cat(emb, dim=0)
    return emb


@torch.no_grad()
def compute_transcripts_batch(audio: list, p: Pipeline):
    """Compute batched transcripts"""

    assert isinstance(p.model, SpeechRecognitionModel)
    transcripts = []
    for i in range(len(audio)):
        t = p.model.transcribe(audio[i].to(p.device))
        if isinstance(t, str):
            transcripts.append(t)
        elif isinstance(t, list):
            transcripts.extend(t)

    assert len(transcripts) == len(audio), f'Transcript format error'

    return transcripts


@torch.no_grad()
def compute_attack_batch(audio: list,
                         a: TrainableAttack,
                         c: torch.Tensor):

    if len(c) < len(audio):
        c = c.repeat(len(audio), 1, 1)
    adv = [a.perturbation(audio[i].to(a.pipeline.device),
                          y=c[i:i+1].to(a.pipeline.device)).to('cpu').reshape(1, 1, -1)
           for i in range(len(audio))]
    return adv


@torch.no_grad()
def compute_pesq(audio1: list, audio2: list, mode: str = 'wb'):

    assert len(audio1) == len(audio2)
    scores = []

    for i in range(len(audio1)):
        try:
            scores.append(
                pesq(DataProperties.get('sample_rate'),
                     tensor_to_np(audio1[i]).flatten(),
                     tensor_to_np(audio2[i]).flatten(),
                     mode)
            )
        except NoUtterancesError:
            print("PESQ error, skipping audio file...")
    return scores


@torch.no_grad()
def compute_stoi(audio1: list, audio2: list, extended: bool = False):

    assert len(audio1) == len(audio2)
    scores = []
    for i in range(len(audio1)):
        scores.append(
            stoi(tensor_to_np(audio1[i]).flatten(),
                 tensor_to_np(audio2[i]).flatten(),
                 DataProperties.get('sample_rate'),
                 extended=extended)
        )
    return scores


@torch.no_grad()
def build_ls_dataset(pipelines: dict):
    """
    Build LibriSpeech evaluation dataset on disk holding:
      * query audio
      * query embeddings
      * conditioning embeddings
      * enrolled embeddings
      * ground-truth query transcripts
    """

    # locate dataset
    data_dir = LIBRISPEECH_DATA_DIR / 'train-clean-360'
    cache_dir = CACHE_DIR / 'ls_wer_eval'
    ensure_dir(cache_dir)

    assert os.path.isdir(data_dir), \
        f'LibriSpeech `train-clean-360` subset required for evaluation'

    spkr_dirs = list(data_dir.glob("*/"))
    spkr_dirs = [s_d for s_d in spkr_dirs if os.path.isdir(s_d)]

    # catalog audio and load transcripts
    for spkr_dir in tqdm(spkr_dirs, total=len(spkr_dirs), desc='Building dataset'):

        # identify speaker
        spkr_id = spkr_dir.parts[-1]

        # check whether cached data exists for speaker
        spkr_cache_dir = cache_dir / spkr_id
        if os.path.isdir(spkr_cache_dir):
            continue

        # each recording session has a separate subdirectory
        rec_dirs = list(spkr_dir.glob("*/"))
        rec_dirs = [r_d for r_d in rec_dirs if os.path.isdir(r_d)]

        # for each speaker, process & store necessary (non-adversarial) data
        all_audio = []
        all_transcripts = []

        # for each recording session, extract all audio files and transcripts
        for rec_dir in rec_dirs:

            rec_id = rec_dir.parts[-1]
            trans_fn = rec_dir / f"{spkr_id}-{rec_id}.trans.txt"

            # open transcript file
            with open(trans_fn, "r") as f:
                trans_idx = f.readlines()

            if len(trans_idx) == 0:
                print(f"Error: empty transcript {trans_fn}")
                continue

            for line in trans_idx:

                split_line = line.strip().split(" ")
                audio_fn = rec_dir / f'{split_line[0]}.{LIBRISPEECH_EXT}'
                transcript = " ".join(split_line[1:]).replace(" ", "|")

                x, _ = li.load(audio_fn, mono=True, sr=16000)
                all_audio.append(torch.as_tensor(x).reshape(1, 1, -1).float())
                all_transcripts.append(transcript)

        # shuffle audio and transcripts in same random order
        all_audio, all_transcripts = shuffle(all_audio, all_transcripts)

        # divide audio and transcripts
        query_audio = all_audio[:N_QUERY]
        query_transcripts = all_transcripts[:N_QUERY]
        condition_audio = all_audio[N_QUERY:N_QUERY+N_CONDITION]
        enroll_audio = all_audio[N_QUERY+N_CONDITION:][:N_ENROLL]

        # check for sufficient audio in each category
        if len(query_audio) < N_QUERY:
            print(f"Error: insufficient query audio for speaker {spkr_id}")
            continue
        elif len(condition_audio) < N_CONDITION:
            print(f"Error: insufficient conditioning audio for speaker {spkr_id}")
            continue
        elif len(enroll_audio) < N_ENROLL:
            print(f"Error: insufficient enrollment audio for speaker {spkr_id}")
            continue

        # compute and save embeddings
        for p_name, p in pipelines.items():

            # compute and save query embeddings
            query_emb = compute_embeddings_batch(query_audio, p)
            f_query = spkr_cache_dir / p_name / 'query_emb.pt'
            ensure_dir_for_filename(f_query)

            # compute and save conditioning embeddings
            condition_emb = compute_embeddings_batch(condition_audio, p)
            f_condition = spkr_cache_dir / p_name / 'condition_emb.pt'
            ensure_dir_for_filename(f_condition)

            # compute and save enrolled embeddings
            enroll_emb = compute_embeddings_batch(enroll_audio, p)
            f_enroll = spkr_cache_dir / p_name / 'enroll_emb.pt'
            ensure_dir_for_filename(f_enroll)

            torch.save(query_emb, f_query)
            torch.save(condition_emb, f_condition)
            torch.save(enroll_emb, f_enroll)

        # save query audio
        f_audio = spkr_cache_dir / 'query_audio.pt'
        torch.save(query_audio, f_audio)

        # save query transcripts
        f_transcript = spkr_cache_dir / 'query_trans.pt'
        torch.save(query_transcripts, f_transcript)

@torch.no_grad()
def build_vc_dataset(pipelines: dict):
    """
    Build VoxCeleb evaluation dataset on disk holding:
      * query audio
      * query embeddings
      * conditioning embeddings
      * enrolled embeddings
    """

    # locate dataset
    data_dir = VOXCELEB1_DATA_DIR / 'voxceleb1'
    cache_dir = CACHE_DIR / 'vc_wer_eval'
    ensure_dir(cache_dir)

    assert os.path.isdir(data_dir), \
        f'VoxCeleb1 dataset required for evaluation'

    spkr_dirs = list(data_dir.glob("*/"))
    spkr_dirs = [s_d for s_d in spkr_dirs if os.path.isdir(s_d)]

    # catalog audio
    for spkr_dir in tqdm(spkr_dirs, total=len(spkr_dirs), desc='Building dataset'):

        # identify speaker
        spkr_id = spkr_dir.parts[-1]

        # check whether cached data exists for speaker
        spkr_cache_dir = cache_dir / spkr_id
        if os.path.isdir(spkr_cache_dir):
            continue

        # each recording session has a separate subdirectory
        rec_dirs = list(spkr_dir.glob("*/"))
        rec_dirs = [r_d for r_d in rec_dirs if os.path.isdir(r_d)]

        # for each speaker, process & store necessary (non-adversarial) data
        all_audio = []

        # for each recording session, extract all audio files and transcripts
        for rec_dir in rec_dirs:
            for audio_fn in rec_dir.glob(f"*.{VOXCELEB1_EXT}"):
                x, _ = li.load(audio_fn, mono=True, sr=16000)
                all_audio.append(torch.as_tensor(x).reshape(1, 1, -1).float())

        # shuffle audio in random order
        all_audio = shuffle(all_audio)

        # divide audio and transcripts
        query_audio = all_audio[:N_QUERY]
        condition_audio = all_audio[N_QUERY:N_QUERY+N_CONDITION]
        enroll_audio = all_audio[N_QUERY+N_CONDITION:][:N_ENROLL]

        # check for sufficient audio in each category
        if len(query_audio) < N_QUERY:
            print(f"Error: insufficient query audio for speaker {spkr_id}")
            continue
        elif len(condition_audio) < N_CONDITION:
            print(f"Error: insufficient conditioning audio for speaker {spkr_id}")
            continue
        elif len(enroll_audio) < N_ENROLL:
            print(f"Error: insufficient enrollment audio for speaker {spkr_id}")
            continue

        # compute and save embeddings
        for p_name, p in pipelines.items():

            # compute and save query embeddings
            query_emb = compute_embeddings_batch(query_audio, p)
            f_query = spkr_cache_dir / p_name / 'query_emb.pt'
            ensure_dir_for_filename(f_query)

            # compute and save conditioning embeddings
            condition_emb = compute_embeddings_batch(condition_audio, p)
            f_condition = spkr_cache_dir / p_name / 'condition_emb.pt'
            ensure_dir_for_filename(f_condition)

            # compute and save enrolled embeddings
            enroll_emb = compute_embeddings_batch(enroll_audio, p)
            f_enroll = spkr_cache_dir / p_name / 'enroll_emb.pt'
            ensure_dir_for_filename(f_enroll)

            torch.save(query_emb, f_query)
            torch.save(condition_emb, f_condition)
            torch.save(enroll_emb, f_enroll)

        # save query audio
        f_audio = spkr_cache_dir / 'query_audio.pt'
        torch.save(query_audio, f_audio)

@torch.no_grad()
def asr_metrics(true: list, hypothesis: list, batch_size: int = 5):
    """
    Compute word and character error rates between two lists of corresponding
    transcripts
    """

    assert len(true) == len(hypothesis)

    n_batches = math.ceil(len(true) / batch_size)

    transform_wer = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveWhiteSpace(replace_by_space=True),
        jiwer.RemoveMultipleSpaces(),
        jiwer.ReduceToSingleSentence(word_delimiter="|"),
        jiwer.ReduceToListOfListOfWords(word_delimiter="|"),
    ])

    wer_score = 0.0
    cer_score = 0.0

    wer_n = 0
    cer_n = 0

    for i in range(n_batches):

        batch_true = true[i*batch_size:(i+1)*batch_size]
        batch_hypothesis = hypothesis[i*batch_size:(i+1)*batch_size]

        wer_n_batch = builtins.sum([len(s.split('|')) for s in batch_true])
        cer_n_batch = builtins.sum([len(s) for s in batch_true])

        attack_cer = cer(batch_true, batch_hypothesis)
        attack_wer = wer(batch_true, batch_hypothesis,
                         truth_transform=transform_wer,
                         hypothesis_transform=transform_wer)

        wer_score += wer_n_batch*attack_wer
        cer_score += cer_n_batch*attack_cer

        wer_n += wer_n_batch
        cer_n += cer_n_batch

    wer_score /= wer_n
    cer_score /= cer_n

    return wer_score, cer_score


@torch.no_grad()
def top_k(query: dict, enrolled: dict, k: int):
    """
    Compute portion of queries for which 'correct' ID appears in k-closest
    enrolled entries
    """

    # concatenate query embeddings into single tensor
    query_array = []
    query_ids = []

    for s_l in query.keys():
        query_array.append(query[s_l])
        query_ids.extend([s_l] * len(query[s_l]))

    query_array = torch.cat(query_array, dim=0).squeeze().cpu().numpy()
    query_ids = torch.as_tensor(query_ids).cpu().numpy()

    # concatenate enrolled embeddings into single tensor
    enrolled_array = []
    enrolled_ids = []

    for s_l in enrolled.keys():
        enrolled_array.append(enrolled[s_l])
        enrolled_ids.extend([s_l] * len(enrolled[s_l]))

    enrolled_array = torch.cat(enrolled_array, dim=0).squeeze().cpu().numpy()
    enrolled_ids = torch.as_tensor(enrolled_ids).cpu().numpy()

    # embedding dimension
    assert query_array.shape[-1] == enrolled_array.shape[-1]
    d = query_array.shape[-1]

    # index enrolled embeddings
    knn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(enrolled_array)

    # `I` is a (n_queries, k) array holding the indices of the k-closest enrolled
    # embeddings for each query; `D` is a (n_queries, k) array holding the corresponding
    # embedding-space distances
    D, I = knn.kneighbors(query_array, k, return_distance=True)

    # for each row, see if at least one of the k nearest enrolled indices maps
    # to a speaker ID that matches the query index's speaker id
    targets = np.tile(query_ids.reshape(-1, 1), (1, k))

    predictions = enrolled_ids[I]
    matches = (targets == predictions).sum(axis=-1) > 0

    return np.mean(matches)


def init_attacks():
    """
    Initialize pre-trained speaker recognition pipelines and de-identification
    attacks
    """

    # channel simulation
    if SIMULATION:
        sim = [
            Offset(length=[-.15, .15]),
            Noise(type='gaussian', snr=[30.0, 50.0]),
            Bandpass(low=[300, 500], high=[3400, 7400]),
            Dropout(rate=0.001)
        ]
    else:
        sim = None

    pipelines = {}

    model_resnet = SpeakerVerificationModel(
        model=ResNetSE34V2(nOut=512, encoder_type='ASP'),
        n_segments=1,
        segment_select='lin',
        distance_fn='cosine',
        threshold=0.0
    )
    model_resnet.load_weights(
        MODELS_DIR / 'speaker' / 'resnetse34v2' / 'resnetse34v2.pt')

    model_yvector = SpeakerVerificationModel(
        model=YVector(),
        n_segments=1,
        segment_select='lin',
        distance_fn='cosine',
        threshold=0.0
    )
    model_yvector.load_weights(
        MODELS_DIR / 'speaker' / 'yvector' / 'yvector.pt')

    pipelines['resnet'] = Pipeline(
        simulation=sim,
        preprocessor=Preprocessor(Normalize(method='peak')),
        model=model_resnet,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    if TRANSFER:
        pipelines['yvector'] = Pipeline(
            simulation=sim,
            preprocessor=Preprocessor(Normalize(method='peak')),
            model=model_yvector,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    else:
        del model_yvector

    # prepare to log attack progress
    writer = Writer(
        root_dir=RUNS_DIR,
        name='evaluate-attacks',
        use_timestamp=True,
        log_iter=300,
        use_tb=True
    )

    attacks = {}

    # use consistent adversarial loss
    adv_loss = SpeakerEmbeddingLoss(
        targeted=False,
        confidence=0.1,
        threshold=0.0
    )

    # use consistent auxiliary loss across attacks
    aux_loss = SumLoss().add_loss_function(
        DemucsMRSTFTLoss(), 1.0
    ).add_loss_function(L1Loss(), 1.0).to('cuda')

    attacks['voicebox'] = VoiceBoxAttack(
        pipeline=pipelines['resnet'],
        adv_loss=adv_loss,
        aux_loss=aux_loss,
        lr=1e-4,
        epochs=1,
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
            'bottleneck_lookahead_frames': LOOKAHEAD,
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
    attacks['voicebox'].load(VOICEBOX_PATH)

    attacks['universal'] = AdvPulseAttack(
        pipeline=pipelines['resnet'],
        adv_loss=adv_loss,
        pgd_norm=float('inf'),
        pgd_variant=None,
        scale_grad=None,
        eps=0.08,
        length=2.0,
        align='start',
        lr=1e-4,
        normalize=True,
        loop=True,
        aux_loss=aux_loss,
        epochs=1,
        batch_size=BATCH_SIZE,
        writer=writer,
        checkpoint_name='universal-attack'
    )
    attacks['universal'].load(UNIVERSAL_PATH)

    attacks['kenansville'] = KenansvilleAttack(
        pipeline=pipelines['resnet'],
        batch_size=BATCH_SIZE,
        adv_loss=adv_loss,
        threshold_db_low=4.0,  # fix threshold
        threshold_db_high=4.0,
        win_length=512,
        writer=writer,
        step_size=1.0,
        search='bisection',
        min_success_rate=0.2,
        checkpoint_name='kenansville-attack'
    )

    attacks['noise'] = WhiteNoiseAttack(
        pipeline=pipelines['resnet'],
        adv_loss=adv_loss,
        aux_loss=aux_loss,
        snr_low=-10.0,  # fix threshold
        snr_high=-10.0,
        writer=writer,
        step_size=1,
        search='bisection',
        min_success_rate=0.2,
        checkpoint_name='noise-perturbation'
    )

    return attacks, pipelines, writer


@torch.no_grad()
def evaluate_attack(attack: TrainableAttack,
                    speaker_pipeline: Pipeline,
                    asr_pipeline: Pipeline):

    if DENOISER:
        from src.models.denoiser.demucs import load_demucs
        defense = load_demucs('dns_48').to(
            'cuda' if torch.cuda.is_available() else 'cpu')
        defense.eval()
    else:
        defense = nn.Identity()

    # prepare for GPU inference
    if torch.cuda.is_available():

        attack.pipeline.set_device('cuda')
        speaker_pipeline.set_device('cuda')
        asr_pipeline.set_device('cuda')
        attack.perturbation.to('cuda')

    # locate dataset
    if EVAL_DATASET == "librispeech":
        cache_dir = CACHE_DIR / 'ls_wer_eval'
    else:
        cache_dir = CACHE_DIR / 'vc_wer_eval'
    assert os.path.isdir(cache_dir), \
        f'Dataset must be built/cached before evaluation'

    # prepare for PESQ/STOI calculations
    all_pesq_scores = []
    all_stoi_scores = []

    # prepare for WER/CER computations
    all_query_transcripts = []
    all_pred_query_transcripts = []
    all_adv_query_transcripts = []

    # prepare for accuracy computations
    all_query_emb = {}
    all_adv_query_emb = {}
    all_enroll_emb = {}
    all_enroll_emb_centroid = {}

    spkr_dirs = list(cache_dir.glob("*/"))
    spkr_dirs = [s_d for s_d in spkr_dirs if os.path.isdir(s_d)]
    for spkr_dir in tqdm(spkr_dirs, total=len(spkr_dirs), desc='Running evaluation'):

        # identify speaker
        spkr_id = spkr_dir.parts[-1]

        # use integer IDs
        if EVAL_DATASET != "librispeech":
            spkr_id = spkr_id.split("id")[-1]

        # identify speaker recognition model
        if isinstance(speaker_pipeline.model.model, ResNetSE34V2):
            model_name = 'resnet'
        elif isinstance(speaker_pipeline.model.model, YVector):
            model_name = 'yvector'
        else:
            raise ValueError(f'Invalid speaker recognition model')

        # load clean embeddings
        query_emb = torch.load(spkr_dir / model_name / 'query_emb.pt')
        condition_emb = torch.load(spkr_dir / 'resnet' / 'condition_emb.pt')
        enroll_emb = torch.load(spkr_dir / model_name / 'enroll_emb.pt')

        # load clean audio
        query_audio = torch.load(spkr_dir / 'query_audio.pt')

        # if defense in use, re-compute query audio
        if DENOISER:
            query_emb = compute_embeddings_batch(
                query_audio, speaker_pipeline, defense=defense
            )

        # load clean transcript
        if EVAL_DATASET == "librispeech":
            query_transcripts = torch.load(spkr_dir / 'query_trans.pt')
        else:
            query_transcripts = None

        # compute conditioning embedding centroid
        condition_centroid = condition_emb.mean(dim=(0, 1), keepdim=True)

        # compute enrolled embedding centroid
        enroll_centroid = enroll_emb.mean(dim=(0, 1), keepdim=True)

        # compute adversarial query audio
        adv_query_audio = compute_attack_batch(
            query_audio, attack, condition_centroid)

        # compute adversarial query embeddings; optionally, pass through
        # unseen denoiser defense
        adv_query_emb = compute_embeddings_batch(
            adv_query_audio, speaker_pipeline, defense=defense
        )

        if EVAL_DATASET == "librispeech":

            # compute clean predicted transcripts
            pred_query_transcripts = compute_transcripts_batch(
                query_audio, asr_pipeline
            )

            # compute adversarial transcripts
            adv_query_transcripts = compute_transcripts_batch(
                adv_query_audio, asr_pipeline
            )

        # compute objective quality metric scores
        if COMPUTE_OBJECTIVE_METRICS:
            pesq_scores = compute_pesq(query_audio, adv_query_audio)
            stoi_scores = compute_stoi(query_audio, adv_query_audio)
        else:
            pesq_scores = np.zeros(len(query_audio))
            stoi_scores = np.zeros(len(query_audio))

        # store all objective quality metric scores
        all_pesq_scores.extend(pesq_scores)
        all_stoi_scores.extend(stoi_scores)

        # store all unit-normalized clean, adversarial, and enrolled centroid
        # embeddings
        all_query_emb[int(spkr_id)] = F.normalize(query_emb.clone(), dim=-1)
        all_adv_query_emb[int(spkr_id)] = F.normalize(adv_query_emb.clone(), dim=-1)
        all_enroll_emb[int(spkr_id)] = F.normalize(enroll_emb.clone(), dim=-1)
        all_enroll_emb_centroid[int(spkr_id)] = F.normalize(enroll_centroid.clone(), dim=-1)

        # store all transcripts
        if EVAL_DATASET == "librispeech":
            all_query_transcripts.extend(query_transcripts)
            all_pred_query_transcripts.extend(pred_query_transcripts)
            all_adv_query_transcripts.extend(adv_query_transcripts)

    # free GPU memory for similarity search
    attack.pipeline.set_device('cpu')
    speaker_pipeline.set_device('cpu')
    asr_pipeline.set_device('cpu')
    attack.perturbation.to('cpu')
    torch.cuda.empty_cache()

    # compute and display final objective quality metrics
    print(f"PESQ (mean/std): {np.mean(all_pesq_scores)}/{np.std(all_pesq_scores)}")
    print(f"STOI (mean/std): {np.mean(all_stoi_scores)}/{np.std(all_stoi_scores)}")

    if EVAL_DATASET == "librispeech":

        # compute and display final WER/CER metrics
        wer, cer = asr_metrics(all_query_transcripts, all_adv_query_transcripts)
        print(f"Adversarial WER / CER: {wer} / {cer}")

        wer, cer = asr_metrics(all_query_transcripts, all_pred_query_transcripts)
        print(f"Clean WER / CER: {wer} / {cer}")

    else:
        wer, cer = None, None

    del (wer, cer, all_pesq_scores, all_stoi_scores,
         all_query_transcripts, all_adv_query_transcripts, all_pred_query_transcripts)

    # embedding-space cosine distance calculations
    cos_dist_fn = EmbeddingDistance(distance_fn='cosine')

    # mean clean-to-adversarial query embedding distance
    total_query_dist = 0.0
    n = 0
    for spkr_id in all_query_emb.keys():
        dist = cos_dist_fn(all_query_emb[spkr_id],
                           all_adv_query_emb[spkr_id]).mean()
        total_query_dist += len(all_query_emb[spkr_id]) * dist.item()
        n += len(all_query_emb[spkr_id])
    mean_query_dist = total_query_dist / n
    print(f"\n\t\tMean cosine distance between clean and adversarial query "
          f"embeddings: {mean_query_dist :0.4f}")

    # mean adversarial-query-to-enrolled-centroid embedding distance
    total_centroid_dist = 0.0
    n = 0
    for spkr_id in all_query_emb.keys():
        n_queries = len(all_adv_query_emb[spkr_id])
        dist = 0.0
        for i in range(n_queries):
            dist += cos_dist_fn(all_enroll_emb_centroid[spkr_id],
                                all_adv_query_emb[spkr_id][i:i+1]).item()
        total_centroid_dist += dist
        n += n_queries
    mean_centroid_dist = total_centroid_dist / n
    print(f"\t\tMean cosine distance between clean enrolled centroids and "
          f"adversarial query embeddings: {mean_centroid_dist :0.4f}")

    # top-1 accuracy for clean queries (closest embedding)
    top_1_clean_single = top_k(all_query_emb, all_enroll_emb, k=1)

    # top-1 accuracy for clean queries (centroid embedding)
    top_1_clean_centroid = top_k(all_query_emb, all_enroll_emb_centroid, k=1)

    # top-10 accuracy for clean queries (closest embedding)
    top_10_clean_single = top_k(all_query_emb, all_enroll_emb, k=10)

    # top-10 accuracy for clean queries (centroid embedding)
    top_10_clean_centroid = top_k(all_query_emb, all_enroll_emb_centroid, k=10)

    # top-1 accuracy for adversarial queries (closest embedding)
    top_1_adv_single = top_k(all_adv_query_emb, all_enroll_emb, k=1)

    # top-1 accuracy for adversarial queries (centroid embedding)
    top_1_adv_centroid = top_k(all_adv_query_emb, all_enroll_emb_centroid, k=1)

    # top-10 accuracy for adversarial queries (closest embedding)
    top_10_adv_single = top_k(all_adv_query_emb, all_enroll_emb, k=10)

    # top-10 accuracy for adversarial queries (centroid embedding)
    top_10_adv_centroid = top_k(all_adv_query_emb, all_enroll_emb_centroid, k=10)

    print(f"\n\t\tTop-1 accuracy (clean embedding / nearest enrolled embedding) {top_1_clean_single :0.4f}",
          f"\n\t\tTop-1 accuracy (clean embedding / nearest enrolled centroid) {top_1_clean_centroid :0.4f}",
          f"\n\t\tTop-10 accuracy (clean embedding / nearest enrolled embedding) {top_10_clean_single :0.4f}"
          f"\n\t\tTop-10 accuracy (clean embedding / nearest enrolled centroid) {top_10_clean_centroid :0.4f}",
          f"\n\t\tTop-1 accuracy (adversarial embedding / nearest enrolled embedding {top_1_adv_single :0.4f}",
          f"\n\t\tTop-1 accuracy (adversarial embedding / nearest enrolled centroid) {top_1_adv_centroid :0.4f}",
          f"\n\t\tTop-10 accuracy (adversarial embedding / nearest enrolled embedding {top_10_adv_single :0.4f}",
          f"\n\t\tTop-10 accuracy (adversarial embedding / nearest enrolled centroid) {top_10_adv_centroid :0.4f}"
          )


@torch.no_grad()
def evaluate_attacks(attacks: dict,
                     speaker_pipelines: dict,
                     asr_pipeline: Pipeline):

    for attack_name, attack in attacks.items():
        for sp_name, sp in speaker_pipelines.items():
            print(f'Evaluating {attack_name} against model {sp_name} '
                  f'{"with" if DENOISER else "without"} denoiser defense')
            evaluate_attack(attack, sp, asr_pipeline)


def main():

    # initial random seed (keep dataset order consistent)
    set_random_seed(0)

    # initialize pipelines
    attacks, pipelines, writer = init_attacks()

    # ensure that necessary data is cached
    if EVAL_DATASET == "librispeech":
        build_ls_dataset(pipelines)
    else:
        build_vc_dataset(pipelines)

    # initialize ASR model
    asr_model = SpeechRecognitionModel(
        model=Wav2Vec2(),
    )
    asr_pipeline = Pipeline(
        model=asr_model,
        preprocessor=Preprocessor(Normalize(method='peak')),
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    writer.log_cuda_memory()

    evaluate_attacks(attacks, pipelines, asr_pipeline)


if __name__ == "__main__":
    main()

