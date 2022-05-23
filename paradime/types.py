import numpy as np
from scipy.sparse import base
import torch

from typing import Collection, Sized, Tuple, Union, Callable, Literal, Any
from nptyping import NDArray, Shape

Diss = Union[
    NDArray[Shape['Dim, Dim'], Any],
    torch.Tensor,
    base.spmatrix,
    Tuple[
        NDArray[Shape['Dim, Nn'], Any],
        NDArray[Shape['Dim, Nn'], Any]
        ]
]

Tensor = Union[
    np.ndarray,
    base.spmatrix,
    torch.Tensor
]

Metric = Union[
    Callable,
    str
]