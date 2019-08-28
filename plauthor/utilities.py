__author__ = 'Shayan Fazeli'
__email__ = 'shayan@cs.ucla.edu'

import os
from typing import Tuple


def separate_path_and_file(filepath: str) -> Tuple[str]:
    """
    This method simply separates the path and filename.
    Parameters
    ----------
    filepath: `str`, required
        The filepath ending in file name and file format
    Returns
    ---------
    (directory, filename): `Tuple[str]`
        It contains the absolute path to the directory, and the filename.
    """
    filepath = os.path.abspath(filepath)
    filename = filepath.split('/')[-1]
    directory = filepath[:-len(filename)]
    return directory, filename
