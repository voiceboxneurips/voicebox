#!/bin/bash

set -e

DOWNLOADS_SCRIPTS_DIR=$(eval dirname "$(readlink -f "$0")")
SCRIPTS_DIR="$(dirname "$DOWNLOADS_SCRIPTS_DIR")"
PROJECT_DIR="$(dirname "$SCRIPTS_DIR")"

DATA_DIR="${PROJECT_DIR}/data/"
CACHE_DIR="${PROJECT_DIR}/cache/"

mkdir -p "${DATA_DIR}"
mkdir -p "${CACHE_DIR}"

# download train-clean-360 subset
echo "downloading LibriSpeech train-clean-360..."
wget http://www.openslr.org/resources/12/train-clean-360.tar.gz

# extract train-clean-360 subset
echo "extracting LibriSpeech train-clean-360..."
tar -xf train-clean-360.tar.gz \
    -C "${DATA_DIR}"

# delete archive
rm -f "train-clean-360.tar.gz"
