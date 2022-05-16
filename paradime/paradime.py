from datetime import datetime
import warnings
from typing import Union, Callable, Literal
# from grpc import Call
from numba.core.types.scalars import Boolean
import torch
import torch.utils.data as td
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numba import jit
from scipy.sparse import base
from scipy.optimize import root_scalar
from pynndescent import NNDescent
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

from .dissimilarity import Dissimilarity
from .types import Tensor

class Dataset(td.Dataset ):
    pass

class Loss():
    pass

class Sampler():
    pass


class Encoder(nn.Module):
    pass


class Decoder(nn.Module):
    pass


class ParametricDR():

    def __init__(self,
        dataset: Dataset,
        sampler: Sampler,
        encoder: Encoder,
        decoder: Decoder,
        loss_function: Union[Callable, Loss],
        high_dim_distance: Dissimilarity,
        low_dim_distance: Dissimilarity
        ) -> None:

        self.dataset = dataset
        self.sampler = sampler
        self.encoder = encoder
        self.decoder = decoder
        self.trained = False

        pass

    def __call__(self,
        X: Tensor
        ) -> torch.Tensor: 
        
        return self.encode(X)

    def encode(self,
        X: Tensor
        ) -> torch.Tensor:

        if self.trained:
            return self.encoder(X)
        else:
            raise Exception("Encoder not trained!")

    def train(self):
        pass


def _convert_input_to_numpy(
    X: Tensor) -> np.ndarray:
    
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    elif isinstance(X, base.spmatrix):
        X = X.toarray()
    elif isinstance(X, np.ndarray):
        pass
    else:
        raise Exception(f'Input type {type(X)} not supported')

    return X

def _convert_input_to_torch(
    X: Tensor) -> torch.Tensor:

    if isinstance(X, torch.Tensor):
        pass
    elif isinstance(X, base.spmatrix):
        # TODO: conserve sparseness
        X = torch.tensor(X.toarray())
    elif isinstance(X, np.ndarray):
        X = torch.tensor(X)
    else:
        raise Exception(f'Input type {type(X)} not supported')

    return X
