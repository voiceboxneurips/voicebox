import math
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import hashlib

from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence

from typing import Union, List, Sized, Iterable

from tqdm import tqdm

from src.utils.filesystem import ensure_dir
from src.utils.device import DataParallelWrapper
from src.pipelines import Pipeline
from src.models.speaker.speaker import SpeakerVerificationModel
from src.models.speech.speech import SpeechRecognitionModel
from src.constants import *

################################################################################
# Data-loading utilities
################################################################################


class DatasetWrapper(Dataset):
    """
    Most data utilities here involve re-assigning or computing targets to
    train or evaluate adversarial attacks. This class wraps an existing
    dataset to overwrite its stored inputs and targets as necessary.
    """
    def __init__(self, dataset, inputs, targets):

        super().__init__()

        self.dataset = dataset
        self.inputs = inputs
        self.targets = targets

        ref_batch = next(iter(dataset))

        if isinstance(ref_batch, tuple):
            self.format = 'tuple'
        else:
            self.format = 'dict'

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        if self.format == 'tuple':
            x, y, *features = self.dataset[idx]
            return self.inputs[idx], self.targets[idx], *features

        else:
            batch = self.dataset[idx]
            batch['x'] = self.inputs[idx]
            batch['y'] = self.targets[idx]
            return batch


def pad_batch_power_2(batch):
    """
    Given a batch of tensors, pad to nearest power of 2 to maximum length. Used
    as a `collate_fn` argument to Pytorch `DataLoader` objects.

    Parameters:
    -----------
    batch

    Returns:
    --------
    """

    # get tensors
    (x, y) = zip(*batch)

    n_batch = len(x)

    if n_batch < 1:
        return torch.Tensor([]), None

    if n_batch == 1:
        return x[0:1], y

    if type(y[0]) != str:
        y = torch.stack(y, dim=0)

    # compute maximum length
    dtype, device = x[0].dtype, x[0].device
    lengths = [x_i.shape[-1] for x_i in x]
    max_len = max(lengths)
    next_pow_2 = 2**(max_len - 1).bit_length()

    # pad inputs
    shape = next(iter(x)).shape[1:-1]
    batch_padded = torch.zeros(
        (n_batch, *shape, next_pow_2),
        dtype=dtype,
        device=device
    )

    for i in range(n_batch):
        batch_padded[i, ..., :lengths[i]] = x[i]

    return batch_padded, y


def text_to_tensor(
        text: Union[str, List[str]],
        labels: list,
        return_lengths: bool = True,
        max_length: int = None,
        padding_value: int = -1):
    """
    Convert one or more string transcripts to padded tensor form (character
    indices), and optionally return sequence lengths.

    Parameters:
    -----------
    text (str):            a string or list of string transcripts

    labels (list):         list of characters, ordered by index

    return_lengths (bool): if True, return sequence lengths

    max_length (int):      if given, trim/pad all sequences to length

    padding_value (int):   value with which to perform length padding

    Returns:
    --------
    sequences (Tensor): tensor containing padded index sequences

    lengths (Tensor):   tensor containing sequence lengths
    """

    if isinstance(text, str):
        text = [text]

    # convert from characters to token indices
    char_to_idx = {labels[i].upper(): i for i in range(len(labels))}

    lengths = []
    tensors = []

    for t in text:

        lengths.append(len(t))
        token_indices = [char_to_idx[c] for c in t.upper()]
        tensors.append(
            torch.as_tensor(token_indices, dtype=torch.long)
        )

    # pad and return
    tensors = pad_sequence(
        tensors,
        batch_first=True,
        padding_value=padding_value
    )  # (n_batch, max_seq_len)

    if max_length is not None:
        if tensors.shape[-1] > max_length:
            tensors = tensors[..., :max_length]
        elif tensors.shape[-1] < max_length:
            tensors = F.pad(
                tensors,
                (0, max_length - tensors.shape[-1]),
                value=padding_value)

    lengths = torch.as_tensor(lengths, dtype=torch.long)

    if return_lengths:
        return tensors, lengths
    else:
        return tensors


def padded_transcript_length(
        transcript: torch.Tensor,
        padding_value: int = -1):
    """
    Given one or more transcripts in index sequence format, determine lengths
    by looking for padding value.

    Parameters:
    -----------
    transcript (Tensor): tensor containing one or more index sequences

    padding_value (int): value used to pad sequence tensors

    Returns:
    --------
    lengths (Tensor): tensor containing un-padded length of each sequence
    """

    # find first occurence of padding value in each transcript tensor
    mask = transcript == padding_value
    mask_max_values, mask_max_indices = torch.max(mask, dim=-1)

    # if the max-mask is zero, there is no padding in the tensor
    mask_max_indices[mask_max_values == 0] = transcript.shape[-1]

    return mask_max_indices.long()


def move_to_device_recursive(d: dict, device: Union[str, torch.device]):
    """Move all tensors in a dictionary object to given device"""
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = move_to_device_recursive(v, device)
        elif isinstance(v, torch.Tensor):
            d[k] = v.to(device)
        elif isinstance(v, tuple) and len(v) > 0 and isinstance(
                next(iter(v)), torch.Tensor):
            v_new = tuple()
            for v_i in v:
                v_new += (v_i.to(device),)
            d[k] = v_new
        elif isinstance(v, list) and len(v) > 0 and isinstance(
                next(iter(v)), torch.Tensor):
            v_new = []
            for v_i in v:
                v_new += [v_i.to(device)]
            d[k] = v_new
    return d


def dataset_to_device(data: Dataset, device: Union[str, torch.device]):
    """Move datasets directly to given device. May cause memory issues!"""

    data.__dict__ = move_to_device_recursive(data.__dict__, device)


@torch.no_grad()
def create_embedding_dataset(data_train: Dataset,
                             data_test: Dataset,
                             pipeline: Pipeline,
                             select_train: str = 'random',
                             select_test: str = 'random',
                             targeted: bool = True,
                             target_class: int = None,
                             num_per_class_train: int = None,
                             num_per_class_test: int = None,
                             num_embeddings_train: int = 5,
                             exclude_class: Union[int, List] = None,
                             exclude_success: bool = False,
                             use_cache: bool = False,
                             shuffle: bool = False,
                             **kwargs
                             ):
    """
    Given training and test datasets holding audio and labels, compute
    embeddings according to (potentially reassigned) labels. It is assumed
    that the provided train and test targets both index the same set of classes,
    i.e. that label 0 in the train set refers to the same class as label 0 in
    the test set. This function considers three possible cases:

      1. `targeted` == True, `target_class` != None

         In this case, a single target class from the training set is assigned
         to all training and test instances. Embeddings corresponding to the
         target class are computed and assigned based on the `select` parameter.

      2.  `targeted` == True, `target_class` == None

         In this case, targets are randomly reassigned within both the training
         and test sets. Embeddings corresponding to the targets are computed and
         assigned based on the `select` parameter.

      3.  `targeted` == False

         In this case, targets remain unchanged and the embedding of each
         instance is computed directly.

    Parameters
    ----------
    data_train (Dataset):       a Dataset object holding audio and labels

    data_test (Dataset):        a Dataset object holding audio and labels

    pipeline (Pipeline):        a Pipeline object; must wrap a
                                SpeakerVerificationModel object

    select_train (str):         method of selecting target embeddings for train
                                set; must be one of `random`, `single`, 'same',
                                `centroid`, or 'median'

    select_test (str):          method of selecting target embeddings for test
                                set; must be one of `random`, `single`, same',
                                `centroid`, or 'median'

    targeted (bool):            if False, target classes will not be reassigned

    target_class (int):         if given, reassign all targets to this class

    num_per_class_train (int):  if given, perform stratified sampling with this
                                number of instances drawn per class

    num_per_class_test (int):   if given, perform stratified sampling with this
                                number of instances drawn per class

    num_embeddings_train (int): if given and attack is targeted, train on only
                                this many distinct embeddings of the target
                                speaker, and evaluate on other embeddings of the
                                target speaker

    exclude_class (int, list):  if given, exclude all instances from this class
                                or in this list of classes

    exclude_success (bool):     if True, drop or replace instances for which the
                                initial prediction achieves the desired
                                adversarial outcome (i.e. matches the target in
                                the case of a targeted attack, and evades the
                                target in the case of an untargeted attack)

    use_cache (bool):           if True, save to and load from disk using hash-
                                based lookup

    shuffle (bool):             if True, shuffle data and targets; may result in
                                mismatch if dataset contains other features (e.g.
                                pitch, periodicity)

    Returns
    -------
    train (dict): dictionary containing Dataset with audio and embedding
                  targets, example audio of each target class, original targets
                  indices, and reassigned target indices

    test (dict):  dictionary containing Dataset with audio and embedding
                  targets, example audio of each target class, original targets
                  indices, and reassigned target indices
    """

    if not num_per_class_train:
        num_per_class_train = 0
    if not num_per_class_test:
        num_per_class_test = 0

    if targeted and target_class is not None and num_per_class_train:
        assert num_per_class_train > num_embeddings_train, \
            f'For targets drawn from training set, number of embeddings ' \
            f'reserved for training ({num_embeddings_train}) must be less ' \
            f'than number of embeddings computed per class ' \
            f'({num_per_class_train})'

    if exclude_success:
        raise NotImplementedError(f'Target correction not yet implemented; '
                                  f'use `NullAttack` to measure trivial '
                                  f'success rates for now')

    # ensure pipeline is capable of producing embeddings
    assert isinstance(pipeline.model, SpeakerVerificationModel)

    # match devices
    ref_batch_train = next(iter(data_train))
    ref_batch_test = next(iter(data_test))
    if isinstance(ref_batch_train, tuple):
        example_input, *_ = ref_batch_train
    elif isinstance(ref_batch_train, dict):
        example_input = ref_batch_train['x']
    else:
        raise ValueError(f'Dataset must provide batches in tuple or dictionary'
                         f' format')
    orig_device = example_input.device

    # check that model produces embeddings with valid shape
    try:
        embedding_shape = list(
            pipeline.model(example_input.to(pipeline.device)).shape
        )
        assert len(embedding_shape) == 3 and embedding_shape[0] == 1
    except AssertionError:
        raise RuntimeError(f'Speaker verification model must produce '
                           f'embeddings of shape '
                           f'(n_batch, n_segments, embedding_dim)')

    assert isinstance(data_train, Sized) and isinstance(data_test, Sized), \
        f"Datasets must have length attribute accessible via `len()`"

    # check embedding selection method
    assert select_train in ['random', 'single', 'centroid', 'same', 'median'], \
        f"invalid value for `select_train` {select_train}"
    assert select_test in ['random', 'single', 'centroid', 'same', 'median'], \
        f"invalid value for `select_test` {select_test}"

    assert not targeted or select_train != 'same', \
        f'`same` embedding selection only valid for untargeted mode'

    # check for optional `batch_size` argument; otherwise, use batch size of 1
    batch_size = kwargs.get('batch_size', 1)

    # creating embedding datasets is time-consuming; to avoid repeated
    # computation, we can store the generated dataset under a hash
    hash_str = str(pipeline.model)
    hash_str += str(data_train.__class__.__name__)
    hash_str += str(data_test.__class__.__name__)
    hash_str += select_train + select_test
    hash_str += str(targeted) + str(target_class)
    hash_str += str(num_per_class_train) + str(num_per_class_test)
    hash_str += str(exclude_class) + str(exclude_success)

    # obtain hash and convert to filename
    dataset_hash = hashlib.md5(hash_str.encode()).digest()
    dataset_file = str(dataset_hash).replace("\'", "")[1:].replace("\\", ".")
    dataset_file += ".pt"

    # check whether a cached embedding dataset with matching hash exists
    embeddings_cache_dir = Path(CACHE_DIR) / 'embeddings'
    ensure_dir(embeddings_cache_dir)
    cached_datasets = embeddings_cache_dir.glob('*.pt')

    # if dataset is already cached, load and return
    if use_cache and dataset_file in [d.name for d in cached_datasets]:

        dataset = torch.load(embeddings_cache_dir / dataset_file)

        # check for valid dataset structure
        try:
            assert isinstance(dataset, dict)
            assert 'train' in dataset and 'test' in dataset

            return move_to_device_recursive(
                dataset['train'], orig_device
            ), move_to_device_recursive(dataset['test'], orig_device)

        except AssertionError:
            raise RuntimeWarning(f'Invalid dataset structure; will re-compute '
                                 f'and overwrite existing dataset '
                                 f'{dataset_file}')

    # shuffle data
    rand_idx_train = torch.randperm(
        len(data_train)) if shuffle else torch.arange(len(data_train))
    rand_idx_test = torch.randperm(
        len(data_test)) if shuffle else torch.arange(len(data_test))

    # separate data and labels
    if isinstance(ref_batch_train, tuple):
        inputs_train, labels_train, *_ = data_train[:]
    else:
        inputs_train, labels_train = data_train[:]['x'], data_train[:]['y']
    inputs_train_shuffled = inputs_train[rand_idx_train]
    labels_train_shuffled = labels_train[rand_idx_train]

    if isinstance(ref_batch_test, tuple):
        inputs_test, labels_test, *_ = data_test[:]
    else:
        inputs_test, labels_test = data_test[:]['x'], data_test[:]['y']
    inputs_test_shuffled = inputs_test[rand_idx_test]
    labels_test_shuffled = labels_test[rand_idx_test]

    # if target is given, check that it is present in training data
    if target_class is not None:
        assert target_class in labels_train, \
            f'Target class {target_class} is not present in training data'

    # determine train and test labels
    unique_labels_train = [l.item() for l in torch.unique(labels_train)]
    unique_labels_test = [l.item() for l in torch.unique(labels_test)]

    # filter excluded classes (if given) from train and test sets
    if isinstance(exclude_class, List):
        unique_labels_train = [
            l for l in unique_labels_train if l not in exclude_class]
        unique_labels_test = [
            l for l in unique_labels_test if l not in exclude_class]

    elif exclude_class is not None:
        unique_labels_train = [
            l for l in unique_labels_train if not l == exclude_class]
        unique_labels_test = [
            l for l in unique_labels_test if not l == exclude_class]

    # prepare to store one example audio input per label (speaker)
    audio_train = {}
    audio_test = {}

    # prepare to store training and test embeddings by label
    embeddings_train = {}
    embeddings_test = {}

    def compute_embeddings_by_label(
            unique_labels: list,
            inputs: torch.Tensor,
            labels: torch.Tensor,
            saved_audio: dict,
            saved_embeddings: dict,
            num_per_class):
        """
        Compute an embedding for every instance in the given dataset and store
        by label in a dictionary; store one audio example per label in a
        dictionary.
        """

        # compute embeddings over training set and sort by label
        for label in tqdm(
                unique_labels,
                total=len(unique_labels),
                desc="Computing embeddings for dataset"):

            # select training instances of class, allowing for a limit on the
            # number of embeddings stored per class
            x_l = inputs[labels == label]
            n_l = num_per_class if num_per_class else len(x_l)

            # store one audio example per training label
            saved_audio[label] = x_l[0:1]

            # store embeddings per training label
            n_batches = math.ceil(n_l / batch_size)
            saved_embeddings[label] = []
            for i in range(n_batches):
                saved_embeddings[label].append(
                    pipeline.model(
                        x_l[i*batch_size:(i+1)*batch_size].to(pipeline.device)
                    ).to('cpu')  # store intermediate results on CPU
                )

            saved_embeddings[label] = torch.cat(
                saved_embeddings[label], dim=0)[:n_l]

    # compute embeddings over training and test datasets and store by label
    compute_embeddings_by_label(
        unique_labels_train,
        inputs_train_shuffled,
        labels_train_shuffled,
        audio_train,
        embeddings_train,
        num_per_class_train
    )
    compute_embeddings_by_label(
        unique_labels_test,
        inputs_test_shuffled,
        labels_test_shuffled,
        audio_test,
        embeddings_test,
        num_per_class_test
    )

    # filter datasets to remove excluded and target labels
    if targeted and target_class is not None:
        unique_labels_train = [
            l for l in unique_labels_train if not l == target_class]
        unique_labels_test = [
            l for l in unique_labels_test if not l == target_class]

    def reassign_labels(
            unique_labels: list,
            inputs: torch.Tensor,
            labels_orig: torch.Tensor,
            num_per_class):
        """
        Reassign targets, as detailed in documentation above.
        """

        labels_new = torch.full(labels_orig.shape, -1, dtype=labels_orig.dtype)

        # reassign label-by-label
        for i, label in enumerate(
                tqdm(
                    unique_labels,
                    total=len(unique_labels),
                    desc="Reassigning labels for dataset")):

            # select all training instances with label
            idx_l = labels_orig == label
            x_l = inputs[idx_l]

            # store original targets
            y_orig_l = torch.full((len(x_l), ), label)

            # use a placeholder to allow for deletion of rows; overwrite with
            # valid labels and delete rows where -1 remains
            y_new_l = torch.full((len(x_l), ), -1)

            # limit number of instances per class if specified
            n_l = num_per_class if num_per_class else len(x_l)

            # targeted attacks require that the given targets be reassigned
            if targeted:

                # if target class is provided, reassign targets to given class
                if target_class is not None:
                    y_new_l[:n_l] = target_class

                # if no target class is given, randomly reassign targets; ensure
                # that no target is unchanged and new targets are evenly
                # distributed
                else:
                    remaining_labels = [
                        l for l in unique_labels if l != label]

                    for j in range(n_l):
                        y_new_l[j] = random.choice(remaining_labels)

            # otherwise, classes remain unchanged
            else:
                y_new_l[:n_l] = y_orig_l[:n_l]

            # update data and labels, deleting rows corresponding to
            # extraneous inputs (according to `num_per_class`)
            labels_new[idx_l] = y_new_l

        keep_idx = labels_new != -1
        inputs = inputs[keep_idx]
        labels_orig = labels_orig[keep_idx]
        labels_new = labels_new[keep_idx]

        return inputs, labels_orig, labels_new, keep_idx

    # reassign training and test labels if necessary (see documentation
    # above); remove instances of target class and those not required by
    # `num_per_class`, if given
    (
        inputs_train_shuffled,
        labels_train_shuffled,
        labels_train_reassigned,
        select_idx_train
    ) = reassign_labels(
        unique_labels_train,
        inputs_train_shuffled,
        labels_train_shuffled,
        num_per_class_train
    )
    (
        inputs_test_shuffled,
        labels_test_shuffled,
        labels_test_reassigned,
        select_idx_test
    ) = reassign_labels(
        unique_labels_test,
        inputs_test_shuffled,
        labels_test_shuffled,
        num_per_class_test
    )

    # prepare to store target embeddings corresponding to reassigned labels
    embedding_targets_train = torch.empty(
        (len(labels_train_reassigned), *embedding_shape[1:]))
    embedding_targets_test = torch.empty(
        (len(labels_test_reassigned), *embedding_shape[1:]))

    def assign_embeddings(
            labels_new: torch.Tensor,
            embeddings_by_label: dict,
            embedding_targets: torch.Tensor,
            select: str,
            is_train: bool = True
    ):

        # iterate over dataset and associate embedding targets with
        # reassigned labels
        labels_to_assign = [l.item() for l in torch.unique(labels_new)]

        for label in labels_to_assign:

            # find indices for which embeddings of given label are to
            # be assigned
            idx_l = labels_new == label
            n_l = int(torch.sum(idx_l * 1).item())

            if n_l == 0:
                continue

            # obtain all embeddings corresponding to given label
            embeddings_l = embeddings_by_label[label]

            # if untargeted, assign ground-truth embeddings for each instance
            if select == 'same':
                y_emb_l = embeddings_l

            else:

                # separate train and test embeddings of given speaker
                if num_embeddings_train:

                    # for targeted attacks, allow training/testing on separate
                    # small subsets of a speaker's utterances
                    if targeted:
                        assert num_embeddings_train <= len(embeddings_l), \
                            f"`num_embeddings_train` {num_embeddings_train} " \
                            f"is greater than the number of utterances for " \
                            f"speaker {label}"

                        if is_train:
                            embeddings_l = embeddings_l[:num_embeddings_train]
                        else:
                            embeddings_l = embeddings_l[num_embeddings_train:]

                # using `select` parameter, assign embeddings
                y_emb_l = []

                for i in range(n_l):

                    if select == 'single':  # use single embedding
                        y_emb_l.append(embeddings_l[0:1])

                    elif select == 'random':  # use random embeddings
                        emb_idx = random.randint(0, len(embeddings_l) - 1)
                        y_emb_l.append(embeddings_l[emb_idx:emb_idx+1])

                    elif select == 'centroid':  # average over embeddings

                        _, n_segments, embedding_dim = embedding_shape

                        # duplicate over all segments
                        centroid = embeddings_l.mean(dim=(0, 1)).reshape(
                            (1, 1, embedding_dim)
                        ).repeat(1, n_segments, 1)

                        y_emb_l.append(centroid)

                    elif select == 'median':  # median over embeddings

                        _, n_segments, embedding_dim = embedding_shape

                        # duplicate over all segments
                        median = embeddings_l.reshape(
                            n_l*n_segments, -1
                        ).median(dim=0)[0].reshape(
                            (1, 1, embedding_dim)
                        ).repeat(1, n_segments, 1)

                        y_emb_l.append(median)

                    else:
                        raise ValueError(f'Invalid embedding selection method '
                                         f'{select}')

                y_emb_l = torch.cat(y_emb_l, dim=0)

            embedding_targets[idx_l] = y_emb_l

    # with labels finalized, assign embedding targets
    assign_embeddings(
        labels_train_reassigned,
        embeddings_train,
        embedding_targets_train,
        select_train,
        True
    )
    assign_embeddings(
        labels_test_reassigned,
        embeddings_train if targeted and target_class is not None else embeddings_test,
        embedding_targets_test,
        select_test,
        False
    )

    # account for shuffling
    final_idx_train = rand_idx_train[select_idx_train]
    final_idx_test = rand_idx_test[select_idx_test]

    from src.data.dataset import VoiceBoxDataset

    if isinstance(data_train, VoiceBoxDataset):
        data_train_final = data_train.overwrite_dataset(
            inputs_train_shuffled,
            embedding_targets_train,
            final_idx_train
        )
    else:
        data_train_final = DatasetWrapper(
            data_train,
            inputs_train_shuffled,
            embedding_targets_train)

    if isinstance(data_test, VoiceBoxDataset):
        data_test_final = data_test.overwrite_dataset(
            inputs_test_shuffled,
            embedding_targets_test,
            final_idx_test
        )
    else:
        data_test_final = DatasetWrapper(
            data_test,
            inputs_test_shuffled,
            embedding_targets_test)

    # store data and embeddings, audio examples, original targets, and
    # reassigned targets
    train = {
        'dataset': data_train_final,
        'id_to_audio': audio_train,
        'true_id': labels_train_shuffled,
        'target_id': labels_train_reassigned
    }
    test = {
        'dataset': data_test_final,
        'id_to_audio': audio_test,
        'true_id': labels_test_shuffled,
        'target_id': labels_test_reassigned
    }

    if use_cache:
        dataset = {
            'train': train,
            'test': test
        }
        torch.save(dataset, embeddings_cache_dir / dataset_file)

    # restore device and return
    return move_to_device_recursive(
        train, orig_device
    ), move_to_device_recursive(
        test, orig_device
    )


@torch.no_grad()
def create_transcription_dataset(data_train: Dataset,
                                 data_test: Dataset,
                                 pipeline: Pipeline,
                                 targeted: bool = True,
                                 target_transcription: str = None,
                                 output_format: str = 'transcript',
                                 shuffle: bool = False,
                                 **kwargs):
    """
    Given training and test datasets holding audio assign string transcriptions
    for performing speech recognition attacks.

      1. `targeted` == True, `target_transcription` != None

         In this case, a single transcription target is assigned to all instances.

      2.  `targeted` == True, `target_transcription` == None

         In this case, ground-truth transcriptions are randomly reassigned as
         targets within both the training and test sets.

      3.  `targeted` == False

         In this case, ground-truth transcriptions are used as targets.

    Parameters
    ----------
    data_train (Dataset):       a Dataset object holding audio

    data_test (Dataset):        a Dataset object holding audio

    targeted (bool):            if False, target classes will not be reassigned

    target_transcription (str): if given, reassign all targets to the given
                                transcription string

    shuffle (bool):             if True, shuffle data and targets; may result in
                                mismatch if dataset contains other features (e.g.
                                pitch, periodicity)

    Returns
    -------
    train (dict):

    test (dict):
    """

    # check output format (string transcripts or frame-wise token probabilities)
    assert output_format in ['transcript', 'emission'], \
        f'Invalid output format; must be one of `transcript` or `emission`'

    # check for valid model type
    assert isinstance(pipeline.model, SpeechRecognitionModel)

    assert isinstance(data_train, Sized) and isinstance(data_test, Sized), \
        f"Datasets must have length attribute accessible via `len()`"

    # match devices
    ref_batch_train = next(iter(data_train))
    ref_batch_test = next(iter(data_test))
    if isinstance(ref_batch_train, tuple):
        example_input, *_ = ref_batch_train
    elif isinstance(ref_batch_train, dict):
        example_input = ref_batch_train['x']
    else:
        raise ValueError(f'Dataset must provide batches in tuple or dictionary'
                         f' format')

    orig_device = example_input.device

    # check for optional `batch_size` argument; otherwise, use batch size of 1
    batch_size = kwargs.get('batch_size', 1)

    # shuffle data

    rand_idx_train = torch.randperm(
        len(data_train)) if shuffle else torch.arange(len(data_train))
    rand_idx_test = torch.randperm(
        len(data_test)) if shuffle else torch.arange(len(data_test))

    if isinstance(ref_batch_train, tuple):
        inputs_train, *_ = data_train[:]
    else:
        inputs_train = data_train[:]['x']
    inputs_train_shuffled = inputs_train[rand_idx_train]

    if isinstance(ref_batch_test, tuple):
        inputs_test, *_ = data_test[:]
    else:
        inputs_test = data_test[:]['x']
    inputs_test_shuffled = inputs_test[rand_idx_test]

    # if targeted and target transcription provided, simply assign and return
    if targeted and target_transcription is not None:

        assert output_format == 'transcript', \
            f"Target transcript provided; cannot use emission targets"

        # check that target transcript contains character set compatible with
        # pipeline, and does not contain 'blank' character
        valid_characters = deepcopy(pipeline.model.get_labels())
        try:
            del valid_characters[pipeline.model.get_blank_idx()]
        except (IndexError, TypeError):
            pass
        assert all([c in valid_characters for c in target_transcription]), \
            f'Target transcription contains invalid characters'

        single_target = text_to_tensor(
            target_transcription,
            pipeline.model.get_labels(),
            return_lengths=False
        )

        targets_train = single_target.repeat(len(inputs_train_shuffled), 1)
        targets_test = single_target.repeat(len(inputs_test_shuffled), 1)

    # otherwise, compute transcriptions using given pipeline
    else:

        def transcribe(dataset: torch.Tensor):

            results = []

            n_batches = math.ceil(len(dataset) / batch_size)
            for batch_idx in tqdm(
                    range(n_batches),
                    total=n_batches,
                    desc="Computing transcriptions for dataset"):

                x = dataset[
                    batch_idx*batch_size:(batch_idx+1)*batch_size
                    ].to(pipeline.device)

                if output_format == 'transcript':
                    results.extend(pipeline.model.transcribe(x))
                elif output_format == 'emission':
                    results.extend(
                        torch.split(
                            pipeline.model(x).to(orig_device), 1, dim=0))

            if output_format == 'emission':

                # pad to max emission length
                results = pad_sequence(results, batch_first=True).squeeze(1)

            elif output_format == 'transcript':
                results = text_to_tensor(
                    results,
                    pipeline.model.get_labels(),
                    return_lengths=False)

            return results

        targets_train = transcribe(inputs_train_shuffled)
        targets_test = transcribe(inputs_test_shuffled)

        # if targeted, permute transcriptions such that no input retains its
        # original transcription
        if targeted:

            # use derangements with a fixed iteration budget; expected number
            # of iterations required to shuffle with no fixed points is e (~3)
            def derange(x: torch.Tensor):

                max_iter = 10
                orig_shape = x.shape
                x = x.reshape(x.shape[0], -1)

                for i in range(max_iter):

                    rand_idx = torch.randperm(len(x))
                    equal = torch.sum(
                        1.0 * (x == x[rand_idx]),
                        dim=-1
                    ) >= x.shape[-1]

                    if not equal.sum().item():
                        break

                return x[rand_idx].reshape(orig_shape)

            targets_train = derange(targets_train)
            targets_test = derange(targets_test)

    # compute transcript lengths
    if output_format == 'transcript':
        lengths_train = padded_transcript_length(targets_train)
        lengths_test = padded_transcript_length(targets_test)
    elif output_format == 'emission':
        lengths_train = torch.full(
            size=(len(inputs_train_shuffled),),
            fill_value=targets_train.shape[1],
            dtype=torch.long
        )
        lengths_test = torch.full(
            size=(len(inputs_test_shuffled),),
            fill_value=targets_test.shape[1],
            dtype=torch.long
        )
    else:
        raise ValueError(f'Invalid value for `output_format`')

    from src.data.dataset import VoiceBoxDataset

    if isinstance(data_train, VoiceBoxDataset):
        data_train_final = data_train.overwrite_dataset(
            inputs_train_shuffled,
            targets_train,
            rand_idx_train)
    else:
        data_train_final = DatasetWrapper(
            data_train,
            inputs_train_shuffled,
            targets_train)

    if isinstance(data_test, VoiceBoxDataset):
        data_test_final = data_test.overwrite_dataset(
            inputs_test_shuffled,
            targets_test,
            rand_idx_test
        )
    else:
        data_test_final = DatasetWrapper(
            data_test,
            inputs_test_shuffled,
            targets_test)

    train = {
        'dataset': data_train_final,
        'targets': targets_train,
        'target_lengths': lengths_train
    }

    test = {
        'dataset': data_test_final,
        'targets': targets_test,
        'target_lengths': lengths_test
    }

    return move_to_device_recursive(
        train, orig_device
    ), move_to_device_recursive(
        test, orig_device
    )
