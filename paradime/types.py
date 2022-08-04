from typing import Tuple, Union, Callable

import numpy as np
import numpy.typing as npt
import scipy.sparse
import torch

Rels = Union[
    np.ndarray,
    torch.Tensor,
    scipy.sparse.spmatrix,
    Tuple[
        np.ndarray,
        np.ndarray
        ]
]

TensorLike = Union[
    np.ndarray,
    torch.Tensor,
]

BinaryTensorFun = Callable[
    [torch.Tensor, torch.Tensor],
    torch.Tensor
]

IndexList = Union[
    list[int],
    npt.NDArray[np.integer],
    torch.Tensor
]

Data = Union[
    np.ndarray,
    torch.Tensor,
    dict[str, Union[
        np.ndarray,
        torch.Tensor
    ]]
]