import argparse
from pathlib import Path
import subprocess
import hashlib
import tarfile
from zipfile import ZipFile

from src.constants import VOXCELEB1_DATA_DIR, VOXCELEB2_DATA_DIR
from src.utils import ensure_dir

################################################################################
# Download VoxCeleb1 dataset using valid credentials
################################################################################


def parse_args():

    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        '--subset',
        type=int,
        default=1,
        help='Specify which VoxCeleb subset to download: 1 or 2'
    )

    parser.add_argument(
        '--username',
        type=str,
        default=None,
        help='User name provided by VGG to access VoxCeleb dataset'
    )

    parser.add_argument(
        '--password',
        type=str,
        default=None,
        help='Password provided by VGG to access VoxCeleb dataset'
    )

    return parser.parse_args()


def md5(f: str):
    """
    Return MD5 checksum for file. Code adapted from voxceleb_trainer repository:
    https://github.com/clovaai/voxceleb_trainer/blob/master/dataprep.py
    """

    hash_md5 = hashlib.md5()
    with open(f, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download(username: str,
             password: str,
             save_path: str,
             lines: list):
    """
    Given a list of dataset shards formatted as <URL, MD5>, download
    each using `wget` and verify checksums. Code adapted from voxceleb_trainer
    repository:
    https://github.com/clovaai/voxceleb_trainer/blob/master/dataprep.py
    """

    for line in lines:
        url = line.split()[0]
        md5gt = line.split()[1]
        outfile = url.split('/')[-1]

        # download files
        out = subprocess.call(
            f'wget {url} --user {username} --password {password} -O {save_path}'
            f'/{outfile}', shell=True)
        if out != 0:
            raise ValueError(f'Download failed for {url}')

        # verify checksum
        md5ck = md5(f'{save_path}/{outfile}')
        if md5ck == md5gt:
            print(f'Checksum successful for {outfile}')
        else:
            raise Warning(f'Checksum failed for {outfile}')


def concatenate(save_path: str, lines: list):
    """
    Given a specification in the format <FMT, FILENAME, MD5>, concatenate all
    downloaded data shards matching FMT into the file FILENAME and verify
    checksums. Code adapted from voxceleb_trainer repository:
    https://github.com/clovaai/voxceleb_trainer/blob/master/dataprep.py
    """

    for line in lines:
        infile = line.split()[0]
        outfile = line.split()[1]
        md5gt = line.split()[2]

        # concatenate shards
        out = subprocess.call(
            f'cat {save_path}/{infile} > {save_path}/{outfile}', shell=True)

        # verify checksum
        md5ck = md5(f'{save_path}/{outfile}')
        if md5ck == md5gt:
            print(f'Checksum successful for {outfile}')
        else:
            raise Warning(f'Checksum failed for {outfile}')

        # delete shards
        out = subprocess.call(
            f'rm {save_path}/{infile}', shell=True)


def full_extract(save_path: str, f: str):
    """
    Extract contents of compressed archive to data directory
    """

    save_path = str(save_path)
    f = str(f)

    print(f'Extracting {f}')

    if f.endswith(".tar.gz"):
        with tarfile.open(f, "r:gz") as tar:
            tar.extractall(save_path)

    elif f.endswith(".zip"):
        with ZipFile(f, 'r') as zf:
            zf.extractall(save_path)


def main():

    args = parse_args()

    # prepare to load dataset file paths
    downloads_dir = Path(__file__).parent

    if args.subset == 1:
        data_dir = VOXCELEB1_DATA_DIR
    elif args.subset == 2:
        data_dir = VOXCELEB2_DATA_DIR
    else:
        raise ValueError(f'Invalid VoxCeleb subset {args.subset}')

    ensure_dir(data_dir)

    # load dataset file paths
    with open(downloads_dir / f'voxceleb{args.subset}_file_parts.txt', 'r') as f:
        file_parts_list = f.readlines()

    # load output file paths
    with open(downloads_dir / f'voxceleb{args.subset}_files.txt', 'r') as f:
        files_list = f.readlines()

    # download subset
    download(
        username=args.username,
        password=args.password,
        save_path=data_dir,
        lines=file_parts_list
    )

    # merge shards
    concatenate(save_path=data_dir, lines=files_list)

    # account for test data
    archives = [file.split()[1] for file in files_list]
    test = f"vox{args.subset}_test_{'wav' if args.subset == 1 else 'aac'}.zip"
    archives.append(test)

    # extract all compressed data
    for file in archives:
        full_extract(data_dir, data_dir / file)

    # organize extracted data
    out = subprocess.call(f'mv {data_dir}/dev/aac/* {data_dir}/aac/ && rm -r '
                          f'{data_dir}/dev', shell=True)
    out = subprocess.call(f'mv -v {data_dir}/{"wav" if args.subset == 1 else "aac"}/*'
                          f' {data_dir}/voxceleb{args.subset}', shell=True)


if __name__ == "__main__":
    main()
