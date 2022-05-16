import numpy as np
from scipy.sparse import base
import torch

from typing import Union, Callable, Literal

Tensor = Union[
    np.ndarray,
    base.spmatrix,
    torch.Tensor
]

Metric = Union[
    Callable,
    str
]