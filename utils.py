import os
import pickle
from pathlib import Path
from inspect import stack
import numpy as np


def to_path(path, depth=5):
    root_path = Path('.').resolve()
    cur_depth = str(root_path).count(os.sep)
    while cur_depth != depth:
        root_path = root_path.parent
        cur_depth = str(root_path).count(os.sep)

    return Path(os.path.join(root_path, path))


def create_directory(path, path_is_directory=False):
    p = to_path(path)
    if not path_is_directory:
        dirname = p.parent
    else:
        dirname = p
    if not dirname.exists():
        os.makedirs(dirname)


def pickle_dump(obj, filename):
    create_directory(filename)
    with open(to_path(filename), 'wb') as f:
        pickle.dump(obj, f)


def pickle_load(filename):
    with open(to_path(filename), 'rb') as f:
        obj = pickle.load(f)

    return obj