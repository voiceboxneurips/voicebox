import os
from pathlib import Path

from typing import Union

################################################################################
# Filesystem utilities
################################################################################


def ensure_dir_for_filename(filename: str):
    """
    Ensure all directories along given path exist, given filename
    """
    ensure_dir(os.path.dirname(filename))


def ensure_dir(directory: Union[str, Path]):
    """
    Ensure all directories along given path exist, given directory name
    """

    directory = str(directory)

    if len(directory) > 0 and not os.path.exists(directory):
        os.makedirs(directory)
