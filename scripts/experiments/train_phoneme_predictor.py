import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.phoneme import PPGEncoder
from src.constants import LIBRISPEECH_NUM_PHONEMES, LIBRISPEECH_PHONEME_DICT
from src.data import LibriSpeechDataset
from src.utils.writer import Writer

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

################################################################################
# Train a simple model to produce phonetic posteriorgrams (PPGs)
################################################################################


def main():

    # training hyperparameters
    lr = .001
    epochs = 60
    batch_size = 250
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # phoneme encoder hyperparameters
    lstm_depth = 2
    hidden_size = 128  # 512
    win_length = 256
    hop_length = 128
    n_mels = 32
    n_mfcc = 19
    lookahead_frames = 0  # 1

    # datasets and loaders
    train_data = LibriSpeechDataset(
        split='train-clean-100',
        target='phoneme',
        features=None,
        hop_length=hop_length
    )
    val_data = LibriSpeechDataset(
        split='test-clean',
        target='phoneme',
        features=None,
        hop_length=hop_length
    )
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True)
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size)

    # initialize phoneme encoder
    encoder = PPGEncoder(
        win_length=win_length,
        hop_length=hop_length,
        win_func=torch.hann_window,
        n_mels=n_mels,
        n_mfcc=n_mfcc,
        lstm_depth=lstm_depth,
        hidden_size=hidden_size,
    )

    # initialize classification layer and wrap as single module
    classifier = nn.Sequential(
        encoder,
        nn.Linear(hidden_size, LIBRISPEECH_NUM_PHONEMES)
    ).to(device)

    # log training progress
    writer = Writer(
        name=f"phoneme_lookahead_{lookahead_frames}",
        use_tb=True,
        log_iter=len(train_loader)
    )

    import builtins
    parameter_count = builtins.sum([
        p.shape.numel()
        for p in classifier[0].parameters()
        if p.requires_grad
    ])

    writer.log_info(f'Training PPG model with lookahead {lookahead_frames}'
                    f' ({parameter_count} parameters)')

    # initialize optimizer and loss function
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    iter_id = 0
    min_val_loss = float('inf')

    for epoch in range(epochs):

        print(f'beginning epoch {epoch}')

        classifier.train()
        for batch in train_loader:

            optimizer.zero_grad(set_to_none=True)

            x, y = batch['x'].to(device), batch['y'].to(device)

            preds = classifier(x)

            # offset labels to incorporate lookahead
            y = y[:, :-lookahead_frames if lookahead_frames else None]

            # offset predictions correspondingly
            preds = preds[:, lookahead_frames:]

            # compute cross-entropy loss
            loss = loss_fn(
                preds.reshape(-1, LIBRISPEECH_NUM_PHONEMES), y.reshape(-1)
            )

            loss.backward()
            optimizer.step()

            writer.log_scalar(loss, tag="CrossEntropyLoss-Train", global_step=iter_id)
            iter_id += 1

        val_loss, val_acc, n = 0.0, 0.0, 0
        classifier.eval()
        with torch.no_grad():
            for batch in val_loader:

                x, y = batch['x'].to(device), batch['y'].to(device)

                preds = classifier(x)

                # offset labels to incorporate lookahead
                y = y[:, :-lookahead_frames if lookahead_frames else None]

                # offset predictions correspondingly
                preds = preds[:, lookahead_frames:]

                n += len(x)
                val_loss += loss_fn(
                    preds.reshape(-1, LIBRISPEECH_NUM_PHONEMES), y.reshape(-1)
                ) * len(x)
                val_acc += len(x) * (torch.argmax(preds, dim=2) == y).flatten().float().mean()

        val_loss /= n
        val_acc /= n
        writer.log_scalar(val_loss, tag="CrossEntropyLoss-Val", global_step=iter_id)
        writer.log_scalar(val_acc, tag="Accuracy-Val")

        # save weights
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            print(f'new best val loss {val_loss}; saving weights')
            writer.checkpoint(classifier[0].state_dict(), 'phoneme_classifier')

    # generate confusion matrix
    classifier.eval()

    # compute accuracy on validation data
    all_preds = []
    all_true = []
    with torch.no_grad():
        for batch in val_loader:

            x, y = batch['x'].to(device), batch['y'].to(device)

            preds = classifier(x)

            # offset labels to incorporate lookahead
            y = y[:, :-lookahead_frames if lookahead_frames else None]

            # offset predictions correspondingly
            preds = preds[:, lookahead_frames:]

            all_preds.append(preds.argmax(dim=2).reshape(-1))
            all_true.append(y.reshape(-1))

    # compile predictions and targets
    all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
    all_true = torch.cat(all_true, dim=0).cpu().numpy()

    reverse_dict = {v: k for (k, v) in LIBRISPEECH_PHONEME_DICT.items() if v != 0}
    reverse_dict[0] = 'sil'

    class_report = classification_report(all_true, all_preds)
    writer.log_info(class_report)

    cm = confusion_matrix(all_true, all_preds, labels=list(range(len(reverse_dict))))
    df_cm = pd.DataFrame(cm, index=[i for i in sorted(list(reverse_dict.keys()))],
                         columns=[i for i in sorted(list(reverse_dict.keys()))])
    plt.figure(figsize=(40, 28))
    sn.set(font_scale=1.0)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 35 / np.sqrt(len(cm))}, fmt='g')

    plt.savefig("phoneme_cm.png", dpi=200)


if __name__ == '__main__':
    main()
