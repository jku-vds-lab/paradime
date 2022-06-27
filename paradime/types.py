import numpy as np
from scipy.sparse import spmatrix
import torch

from typing import Collection, Sized, Tuple, Union, Callable, Literal, Any

Diss = Union[
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

Symm = Union[
    None,
    Literal['tsne', 'umap'],
    Callable[
        [Diss],
        Diss
        ]
]