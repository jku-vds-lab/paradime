"""Type definitions for ParaDime."""

from typing import Callable, Literal, Tuple, Union

import numpy as np
import numpy.typing as npt
import scipy.sparse
import torch

Rels = Union[
    np.ndarray,
    torch.Tensor,
    scipy.sparse.spmatrix,
    Tuple[np.ndarray, np.ndarray],
]

TensorLike = Union[
    np.ndarray,
    torch.Tensor,
]

TypeKeyTuples = list[tuple[Union[Literal["data"], Literal["rels"]], str]]

BinaryTensorFun = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

IndexList = Union[list[int], npt.NDArray[np.integer], torch.Tensor]
