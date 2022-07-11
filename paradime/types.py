import numpy as np
import numpy.typing as npt
from scipy.sparse import spmatrix
import torch

from typing import Collection, Sized, Tuple, Union, Callable, Literal, Any

Rels = Union[
    np.ndarray,
    torch.Tensor,
    spmatrix,
    Tuple[
        np.ndarray,
        np.ndarray
        ]
]

Tensor = Union[
    np.ndarray,
    spmatrix,
    torch.Tensor
]

Metric = Union[
    Callable,
    str
]

LossFun = Callable[
    [torch.Tensor, torch.Tensor],
    torch.Tensor
]

IndexList = Union[
    list[int],
    npt.NDArray[np.integer],
    torch.Tensor
]