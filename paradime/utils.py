"""Utility functions for paraDime.

The :mod:`paradime.utils` module implements various utility functions and
classes, such as a mixin for representations, a rporting method, and input
conversion methods.
"""

from datetime import datetime
import functools
import random
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
    """Prints a timestamp followed by the given message.
    
    Args:
        message: The message string to print.    
    """
    
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

def seed_all(seed:int) -> torch.Generator:
    """Sets several seeds to maximize reproducibility.

    For infos on reproducibility in PyTorch, see
    https://pytorch.org/docs/stable/notes/randomness.html.
    
    Args:
        seed: The integer to use as a seed.

    Returns:
        The :class:`torch.Generator` instance returned by
        :func:`torch.manual_seed`.
    """
    random.seed(seed)
    np.random.seed(seed)
    gen = torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    return gen
    
def get_color_palette() -> dict[str, str]:
    """Get the custom paraDime color palette.
    
    The palette is usually located in an assets folder in the form of a JSON
    file. If the JSON file is not found, this method attemps to create it from
    parsing an SVG file.

    Returns:
        The color palette as a dict of names and hex color values.

    Raises:
        FileNotFoundError: If neither the JSON nor the SVG file can be found.
    """
    import json
    import os
    import sys

    utils_path = os.path.dirname(__file__)
    assets_path = os.path.abspath(os.path.join(utils_path, '../assets'))
    json_path = os.path.join(assets_path, 'palette.json')
    svg_path = os.path.join(assets_path, 'palette.svg')

    if not os.path.isfile(json_path):
        if os.path.isfile(svg_path):
            sys.path.append(os.path.join(assets_path))
            from make_palette import make_palette # type:ignore
            make_palette(svg_path, json_path)
        else:
            raise FileNotFoundError(
                "Could not find JSON or SVG file to create/import palette."
            )
    with open(json_path, 'r') as f:
            return json.load(f)
