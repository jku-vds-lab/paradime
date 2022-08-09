"""Conversion utilities for paraDime.

The :mod:`paradime.utils.convert` module implements various conversion
functions for tensors-like objects and index lists.
"""

import functools
from typing import Union

import numpy as np
import torch

from paradime.types import TensorLike

def to_numpy(X: Union[TensorLike, list[float]]) -> np.ndarray:
    """Converts a tensor-like object to a NumPy array.
    
    Args:
        X: The tensor-like object to be converted.
    
    Returns:
        The resulting Numpy array.
    """
    if isinstance(X, torch.Tensor):
        return X.detach().cpu().numpy()
    elif isinstance(X, np.ndarray):
        return X
    elif isinstance(X, list):
        return np.array(X)
    else:
        raise TypeError(f"Input type {type(X)} not supported")



def to_torch(X: Union[TensorLike, list[float]]) -> torch.Tensor:
    """Converts a tensor-like object to a PyTorch tensor.
    
    Args:
        X: The tensor-like object to be converted.
    
    Returns:
        The resulting PyTorch tensor.
    """
    if isinstance(X, torch.Tensor):
        return X
    elif isinstance(X, (np.ndarray, list)):
        return torch.tensor(X, dtype=torch.float)
    else:
        raise TypeError(f"Input type {type(X)} not supported")

@functools.cache
def rowcol_to_triu_index(i: int, j: int, dim: int) -> int:
    """Converts matrix indices to upper-triangular form.
    
    Converts a pair of row and column indices of a symmetrical square array to
    the corresponding index of the list of upper triangular values.

    Args:
        i: The row index.
        j: The column index.
        dim: The size of the square matrix.

    Returns:
        The upper triangular index.

    Raises:
        ValueError: For diagonal indices (i.e., if i equals j).
    """
    if i < j:
        index = round(i * (dim - 1.5) + j - i**2 * 0.5 - 1)
        return index
    elif i > j:
        return rowcol_to_triu_index(j, i, dim)
    else:
        raise ValueError(
            "Indices of diagonal elements cannot be converted to "
            "upper-triangular form."
        )

@functools.cache
def triu_to_square_dim(len_triu: int) -> int:
    """Calculates the size of a square matrix given the length of the list of
    its upper-triangular values.

    Args:
        len_triu: The lenght of the list of upper-triangular values.

    Returns:
        The size of the square matrix.
    """
    return int(np.ceil(np.sqrt(len_triu * 2)))
