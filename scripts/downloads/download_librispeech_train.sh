#!/bin/bash

set -e

DOWNLOADS_SCRIPTS_DIR=$(eval dirname "$(readlink -f "$0")")
SCRIPTS_DIR="$(dirname "$DOWNLOADS_SCRIPTS_DIR")"
PROJECT_DIR="$(dirname "$SCRIPTS_DIR")"

DATA_DIR="${PROJECT_DIR}/data/"
CACHE_DIR="${PROJECT_DIR}/cache/"

mkdir -p "${DATA_DIR}"
mkdir -p "${CACHE_DIR}"

# download test-clean subset
echo "downloading LibriSpeech test-clean..."
wget http://www.openslr.org/resources/12/test-clean.tar.gz

# extract test-clean subset
echo "extracting LibriSpeech test-clean..."
tar -xf test-clean.tar.gz \
    -C "${DATA_DIR}"

# delete archive
rm -f "test-clean.tar.gz"

# download test-other subset
echo "downloading LibriSpeech test-other..."
wget http://www.openslr.org/resources/12/test-other.tar.gz

# extract test-other subset
echo "extracting LibriSpeech test-other..."
tar -xf test-other.tar.gz \
    -C "${DATA_DIR}"

# delete archive
rm -f "test-other.tar.gz"

# download train-clean-100 subset
echo "downloading LibriSpeech train-clean-100..."
wget http://www.openslr.org/resources/12/train-clean-100.tar.gz

# extract train-clean-100 subset
echo "extracting LibriSpeech train-clean-100..."
tar -xf train-clean-100.tar.gz \
    -C "${DATA_DIR}"

# delete archive
rm -f "train-clean-100.tar.gz"

# download LibriSpeech alignments dataset
wget -O alignments.zip https://zenodo.org/record/2619474/files/librispeech_alignments.zip?download=1
unzip -d "${DATA_DIR}/LibriSpeech/" alignments.zip
rm  -f alignments.zip
