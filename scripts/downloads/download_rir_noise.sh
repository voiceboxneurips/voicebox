#!/bin/bash

set -e

DOWNLOADS_SCRIPTS_DIR=$(eval dirname "$(readlink -f "$0")")
SCRIPTS_DIR="$(dirname "$DOWNLOADS_SCRIPTS_DIR")"
PROJECT_DIR="$(dirname "$SCRIPTS_DIR")"

DATA_DIR="${PROJECT_DIR}/data/"
CACHE_DIR="${PROJECT_DIR}/cache/"

REAL_RIR_DIR="${DATA_DIR}/rir/real/"
SYNTHETIC_RIR_DIR="${DATA_DIR}/rir/synthetic/"
ROOM_NOISE_DIR="${DATA_DIR}/noise/room/"
PS_NOISE_DIR="${DATA_DIR}/noise/pointsource/"

mkdir -p "${REAL_RIR_DIR}"
mkdir -p "${SYNTHETIC_RIR_DIR}"
mkdir -p "${ROOM_NOISE_DIR}"
mkdir -p "${PS_NOISE_DIR}"

# download RIR/noise composite dataset
echo "downloading RIR/noise dataset..."
wget -O "${DATA_DIR}/rirs_noises.zip" https://www.openslr.org/resources/28/rirs_noises.zip

# extract RIR/noise composite dataset
echo "unzipping RIR/noise dataset..."
unzip "${DATA_DIR}/rirs_noises.zip" -d "${DATA_DIR}/"

# delete archive
rm -f "${DATA_DIR}/rirs_noises.zip"

# organize pointsource noise data
echo "extracting point-source noise data"
cp -a "${DATA_DIR}/RIRS_NOISES/pointsource_noises"/. "${PS_NOISE_DIR}"

# organize room noise data
echo "extracting room noise data"
room_noises=($(find "${DATA_DIR}/RIRS_NOISES/real_rirs_isotropic_noises/" -maxdepth 1 -name '*noise*' -type f))
cp -- "${room_noises[@]}" "${ROOM_NOISE_DIR}"

# organize real RIR data
echo "extracting recorded RIR data"
rirs=($(find "${DATA_DIR}/RIRS_NOISES/real_rirs_isotropic_noises/" ! -name '*noise*' ))
cp -- "${rirs[@]}" "${REAL_RIR_DIR}"

# organize synthetic RIR data
echo "extracting synthetic RIR data"
cp -a "${DATA_DIR}/RIRS_NOISES/simulated_rirs"/. "${SYNTHETIC_RIR_DIR}"

# delete redundant data
rm -rf "${DATA_DIR}/RIRS_NOISES/"

# separate near-field and far-field RIRs
NEARFIELD_RIR_DIR="${REAL_RIR_DIR}/nearfield/"
FARFIELD_RIR_DIR="${REAL_RIR_DIR}/farfield/"

mkdir -p "${NEARFIELD_RIR_DIR}"
mkdir -p "${FARFIELD_RIR_DIR}"

# read list of far-field RIRs
readarray -t FF_RIR_LIST < "${DOWNLOADS_SCRIPTS_DIR}/ff_rir.txt"

# move far-field RIRs
for name in "${FF_RIR_LIST[@]}"; do
    mv "$name" "${FARFIELD_RIR_DIR}/$(basename "$name")"
done

# move remaining near-field RIRs
for name in "${REAL_RIR_DIR}"/*.wav; do
    mv "$name" "${NEARFIELD_RIR_DIR}/$(basename "$name")"
done

