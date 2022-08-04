"""Utility functions for paraDime.

The :mod:`paradime.utils` module implements various utility functions and
classes, such as a mixin for representations, a rporting method, and input
conversion methods.
"""

from datetime import datetime
import functools
from typing import Union

import numpy as np
import torch

from paradime.types import TensorLike

class _ReprMixin():
    """A mixin implementing a simple __repr__."""

    # based on PyTorch's nn.Module repr
    def __repr__(self) -> str:
        lines = []
        for k, v in self.__dict__.items():
            v_str = repr(v)
            v_str = _addindent(v_str, 2)
            lines.append(f"{k}={v_str},")

        main_str = f"{type(self).__name__}("
        if lines:
            main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str

def _addindent(s: str, num_spaces: int) -> str:
    lines = s.split('\n')
    # don't do anything for single-line stuff
    if len(lines) == 1:
        return s
    first = lines.pop(0)
    lines = [(num_spaces * ' ') + line for line in lines]
    s = '\n'.join(lines)
    s = first + '\n' + s
    return s

def report(message: str) -> None:
    
    now_str = datetime.now().isoformat(
        sep=' ',
        timespec='milliseconds'
    )[:-2]

    print(now_str + ': ' + message)

def _convert_input_to_numpy(X: Union[TensorLike, list[float]]) -> np.ndarray:
    
    if isinstance(X, torch.Tensor):
        return X.detach().cpu().numpy()
    elif isinstance(X, np.ndarray):
        return X
    elif isinstance(X, list):
        return np.array(X)
    else:
        raise TypeError(f"Input type {type(X)} not supported")

def _convert_input_to_torch(X: Union[TensorLike, list[float]]) -> torch.Tensor:

    if isinstance(X, torch.Tensor):
        return X
    elif isinstance(X, (np.ndarray, list)):
        return torch.tensor(X, dtype=torch.float)
    else:
        raise TypeError(f"Input type {type(X)} not supported")

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