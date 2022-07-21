from datetime import datetime
import torch
import numpy as np
import scipy.sparse
import functools
from typing import Union

from paradime.types import Tensor

def report(message: str) -> None:
    
    now_str = datetime.now().isoformat(
        sep=' ',
        timespec='milliseconds'
    )[:-2]

    print(now_str + ': ' + message)

def _convert_input_to_numpy(
    X: Tensor) -> np.ndarray:
    
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    elif isinstance(X, scipy.sparse.spmatrix):
        X = X.toarray()
    elif isinstance(X, np.ndarray):
        pass
    else:
        raise TypeError(f"Input type {type(X)} not supported")

    return X

def _convert_input_to_torch(X: Union[Tensor, list[float]]) -> torch.Tensor:

    if isinstance(X, torch.Tensor):
        pass
    elif isinstance(X, scipy.sparse.spmatrix):
        # TODO: conserve sparseness
        X = torch.tensor(X.toarray())
    elif isinstance(X, (np.ndarray, list)):
        X = torch.tensor(X)
    else:
        raise TypeError(f"Input type {type(X)} not supported")

    return X

@functools.cache
def _rowcol_to_triu_index(i: int, j: int, dim: int) -> int:
    if i < j:
        index = round(i * (dim - 1.5) + j - i**2 * 0.5 - 1)
        return index
    elif i > j:
        return _rowcol_to_triu_index(j, i, dim)
    else:
        return -1

@functools.cache
def _get_orig_dim(len_triu: int) -> int:
    return int(np.ceil(np.sqrt(len_triu * 2)))