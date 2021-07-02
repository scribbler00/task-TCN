__author__ = "Jens Schreiber"
__copyright__ = "Copyright 2017, Intelligent Embedded Systems, Universität Kassel"
__status__ = "Prototype"


from collections.abc import Iterable
from fastcore.basics import listify
import numpy as np
import pandas as pd
import random
import torch

np_int_dtypes = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.intp,
    np.uintp,
]


def get_structure(
    initial_size,
    percental_reduce,
    min_value,
    input_size=None,
    final_outputs=1,
    reverse_structure=False,
):
    """
    Turn the given parameters into the structure of an ann model.

    The 'initial size' acts as the first layer, and each following layer i is of the size
    'initial_size' * (1 - percental_reduce) ^ i. This is repeated until 'min_value' is reached. Finally, 'final_outputs'
    is appended as the last layer.

    Parameters
    ----------
    initial_size : integer
        size of the first layer, and baseline for all following layers.
    percental_reduce : float
        percentage of the size reduction of each subsequent layer.
    min_value : integer
        the minimum layer size up to which the 'initial_size' is used to create new layers.
    input_size : integer
        if not None, a layer of the given size will be prepended to the actual structure.
    final_outputs : integer
        the size of the final layer.

    Returns
    -------
    list
        The finished structure of the ann model.
    """
    ann_structure = [initial_size]
    final_outputs = listify(final_outputs)

    if 0 in final_outputs or (None in final_outputs):
        raise ValueError(
            "Invalid parameters: final_outputs should not contain 0 or None"
        )

    if percental_reduce >= 1.0:
        percental_reduce = percental_reduce / 100.0

    while True:
        new_size = int(ann_structure[-1] - ann_structure[-1] * percental_reduce)

        if new_size <= min_value:
            new_size = min_value
            ann_structure.append(new_size)
            break
        else:
            ann_structure.append(new_size)

    if reverse_structure:
        ann_structure = list(reversed(ann_structure))

    if input_size != None:
        input_size = listify(input_size)
        return input_size + ann_structure + final_outputs

    else:
        return ann_structure + final_outputs


def set_random_states(seed=42):
    """
    Generate a semi-random seed, and pass it to the pytorch framework.
    Parameters
    ----------
    seed : integer
        the initial seed that is used as a basis to generate a new one.

    Returns
    -------

    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
