import math
import torch
import pandas as pd
import numpy as np

from typing import List, Union

from pesq import pesq
from pystoi import stoi
from tqdm import tqdm

from src.data import DataProperties
from src.utils.plotting import tensor_to_np
from src.models.speech import Wav2Vec2, GreedyCTCDecoder

################################################################################
# Utilities for analyzing attack results
################################################################################


@torch.no_grad()
def run_perceptual_evaluation(x: torch.Tensor,
                              x_adv: torch.Tensor,
                              batch_size: int = 1,
                              device: Union[str, torch.cuda.device] = 'cpu',
                              tag: str = None,
                              **kwargs
                              ):
    """
    Compute perceptual quality metrics on pairs of clean and adversarial audio

    Parameters
    ----------
    x (Tensor):        shape

    x_adv (Tensor):    shape

    batch_size (int):

    device (str):

    Returns
    -------


    """

    # check for compatible audio dimensions
    assert x.ndim == x_adv.ndim

    # require batch dimension
    assert x.ndim >= 2
    n_batch = x.shape[0]

    # store results
    results = {}

    # name results
    tag = '' if tag is None else f'{tag}-'

    ############################################################################
    # WAVEFORM P-NORM DISTANCE
    ############################################################################

    # if dimensions match, measure L-2 and L-inf distance between waveforms
    if x.shape == x_adv.shape:

        reduce_dims = tuple(range(1, x.ndim))

        l2 = (x - x_adv).norm(
            p=2, dim=reduce_dims).flatten().tolist()
        linf = (x - x_adv).norm(
            p=float('inf'), dim=reduce_dims).flatten().tolist()

        results = {
            **results,
            tag + 'L2-Waveform': l2,
            tag + 'Linf-Waveform': linf
        }

    ############################################################################
    # PESQ OBJECTIVE MEASURE (DEPRECATED)
    ############################################################################

    assert DataProperties.get('sample_rate') in [16000, 8000], \
        f"Cannot perform PESQ evaluation with sample rate " \
        f"{DataProperties.get('sample_rate')}Hz; must be 8000Hz or 16000Hz"

    wb_scores, nb_scores = [], []

    for i in tqdm(range(n_batch), desc='computing PESQ scores'):

        wb_scores.append(
            pesq(DataProperties.get('sample_rate'),
                 tensor_to_np(x[i]).flatten(),
                 tensor_to_np(x_adv[i]).flatten(),
                 'wb')
        )
        nb_scores.append(
            pesq(DataProperties.get('sample_rate'),
                 tensor_to_np(x[i]).flatten(),
                 tensor_to_np(x_adv[i]).flatten(),
                 'nb')
        )

    results = {
        **results,
        tag + 'PESQ-Wideband': wb_scores,
        tag + 'PESQ-Narrowband': nb_scores,
    }

    ############################################################################
    # STOI OBJECTIVE MEASURE (DEPRECATED)
    ############################################################################

    cl_scores, ex_scores = [], []

    for i in tqdm(range(n_batch), desc='computing STOI scores'):

        cl_scores.append(
            stoi(tensor_to_np(x[i]).flatten(),
                 tensor_to_np(x_adv[i]).flatten(),
                 DataProperties.get('sample_rate'),
                 extended=False)
        )
        ex_scores.append(
            stoi(tensor_to_np(x[i]).flatten(),
                 tensor_to_np(x_adv[i]).flatten(),
                 DataProperties.get('sample_rate'),
                 extended=True)
        )

    results = {
        **results,
        tag + 'STOI-Extended': cl_scores,
        tag + 'STOI-Classical': ex_scores,
    }

    ############################################################################
    # BSS-EVAL SIGNAL METRICS
    ############################################################################

    si_sdr, sd_sdr, snr, srr = [], [], [], []

    for i in tqdm(range(n_batch), desc='computing BSS-EVAL metrics'):

        si_sdr_i, sd_sdr_i, snr_i, srr_i = _bss_eval(
            tensor_to_np(x_adv[i]).flatten(),
            tensor_to_np(x[i]).flatten())

        si_sdr.append(si_sdr_i)
        sd_sdr.append(sd_sdr_i)
        snr.append(snr_i)
        srr.append(srr_i)

    results = {
        **results,
        tag + 'SI-SDR': si_sdr,
        tag + 'SD-SDR': sd_sdr,
        tag + 'SNR': snr,
        tag + 'SRR': srr
    }

    ############################################################################
    # ASR TRANSCRIPTION METRICS
    ############################################################################

    # initialize ASR model / decoder
    model = Wav2Vec2()
    decoder = GreedyCTCDecoder(labels=model.labels)

    # obtain delimiter token
    delimiter = decoder.get_labels()[decoder.get_sep_idx()]

    # move model to given device
    model.to(device)

    # store original and adversarial transcriptions
    transcriptions = []
    transcriptions_adv = []

    n_batches = math.ceil(len(x) / batch_size)
    for i in tqdm(range(n_batches), desc='computing WER/CER'):

        # move batches to device and pass to model
        x_batch = x[batch_size*i:batch_size*(i+1)].to(device)
        x_adv_batch = x_adv[batch_size*i:batch_size*(i+1)].to(device)

        emit_batch = model(x_batch)
        emit_adv_batch = model(x_adv_batch)

        # decode sequence probability emissions to obtain string transcriptions
        transcriptions.extend(decoder(emit_batch)[0])
        transcriptions_adv.extend(decoder(emit_adv_batch)[0])

    # ASR WER
    wer = compute_wer(transcriptions, transcriptions_adv, delimiter)

    # ASR CER
    cer = compute_cer(transcriptions, transcriptions_adv, delimiter)

    results = {
        **results,
        tag + 'ASR-WER': wer,
        tag + 'ASR-CER': cer,
    }

    return results


def compute_wer(
        reference: List[str],
        transcription: List[str],
        delimiter: str = ' '):
    """
    Compute average word error rate (WER) between string transcriptions.

    WER = (Sw + Dw + Iw) / Nw

    where:
      Sw is the number of words substituted,
      Dw is the number of words deleted,
      Iw is the number of words inserted,
      Nw is the number of words in the reference

    Parameters
    ----------

    Returns
    -------

    """

    assert len(reference) == len(transcription)

    # for each reference-transcription pair in batch, count errors of each of
    # the four types as well as total word count

    total_edit_dist = 0
    total_ref_len = 0

    for r, t in zip(reference, transcription):

        edit_dist, ref_len = _word_errors(r, t, delimiter=delimiter)

        if ref_len == 0:
            raise ValueError("Reference sentences must nonzero word count")

        total_edit_dist += edit_dist
        total_ref_len += ref_len

    wer = float(total_edit_dist) / total_ref_len
    return wer


def compute_cer(
        reference: List[str],
        transcription: List[str],
        delimiter: str = ' ',
        remove_delimiter: bool = False):
    """
    Compute average character error rate (CER) between string transcriptions.

    WER = (Sc + Dc + Ic) / Nc

    where:
      Sc is the number of characters substituted,
      Dc is the number of characters deleted,
      Ic is the number of characters inserted,
      Nc is the number of characters in the reference

    Parameters
    ----------

    Returns
    -------

    """

    assert len(reference) == len(transcription)

    # for each reference-transcription pair in batch, count errors of each of
    # the four types as well as total character count

    total_edit_dist = 0
    total_ref_len = 0

    for r, t in zip(reference, transcription):

        edit_dist, ref_len = _char_errors(r,
                                          t,
                                          delimiter,
                                          remove_delimiter)

        if ref_len == 0:
            raise ValueError("Reference sentences must nonzero character count")

        total_edit_dist += edit_dist
        total_ref_len += ref_len

    cer = float(total_edit_dist) / total_ref_len
    return cer


def _word_errors(reference: str, transcription: str, delimiter: str = ' '):
    """
    Compute the Levenshtein distance between reference and transcription
    sequences at word level.
    """

    reference = reference.lower()
    transcription = transcription.lower()

    ref_words = reference.split(delimiter)
    tra_words = transcription.split(delimiter)

    edit_distance = _levenshtein_distance(ref_words, tra_words)
    return float(edit_distance), len(ref_words)


def _char_errors(reference: str,
                 transcription: str,
                 delimiter: str = ' ',
                 remove_delimiter: bool = False
                 ):
    """
    Compute the Levenshtein distance between reference and transcription
    sequences at word level.
    """

    reference = reference.lower()
    transcription = transcription.lower()

    join_char = delimiter
    if remove_delimiter:
        join_char = ''

    reference = join_char.join(filter(None, reference.split(delimiter)))
    transcription = join_char.join(filter(None, transcription.split(delimiter)))

    edit_distance = _levenshtein_distance(reference, transcription)
    return float(edit_distance), len(reference)


def _levenshtein_distance(reference: Union[List[str], str],
                          transcription: Union[List[str], str]):
    """Levenshtein distance is a string metric for measuring the difference
    between two sequences. Informally, the levenshtein disctance is defined as
    the minimum number of single-character edits (substitutions, insertions or
    deletions) required to change one word into the other. We can naturally
    extend the edits to word level when calculate levenshtein disctance for
    two sentences.
    """
    m = len(reference)
    n = len(transcription)

    # special cases
    if reference == transcription:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m
    if m < n:
        reference, transcription = transcription, reference
        m, n = n, m

    # use O(min(m, n)) space
    distance = np.zeros((2, n + 1), dtype=np.int32)

    # initialize distance matrix
    for j in range(0, n + 1):
        distance[0][j] = j

    # calculate Levenshtein distance
    for i in range(1, m + 1):
        prev_row_idx = (i - 1) % 2
        cur_row_idx = i % 2
        distance[cur_row_idx][0] = i
        for j in range(1, n + 1):
            if reference[i - 1] == transcription[j - 1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
            else:
                s_num = distance[prev_row_idx][j - 1] + 1
                i_num = distance[cur_row_idx][j - 1] + 1
                d_num = distance[prev_row_idx][j] + 1
                distance[cur_row_idx][j] = min(s_num, i_num, d_num)

    return distance[m % 2][n]


def _bss_eval(x, x_ref):

    x_ref_energy = (x_ref ** 2).sum()

    alpha = (x_ref @ x / x_ref_energy)

    e_true = x_ref
    e_res = x - e_true

    signal = (e_true ** 2).sum()
    noise = (e_res ** 2).sum()

    snr = 10 * np.log10(signal / noise)

    e_true = x_ref * alpha
    e_res = x - e_true

    signal = (e_true ** 2).sum()
    noise = (e_res ** 2).sum()

    si_sdr = 10 * np.log10(signal / noise)

    srr = -10 * np.log10((1 - (1/alpha)) ** 2)
    sd_sdr = snr + 10 * np.log10(alpha ** 2)

    return si_sdr, sd_sdr, snr, srr
