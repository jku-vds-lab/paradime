import datetime
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

# type alias for tensor types
Tensor = Union[
    np.array,
    base.spmatrix,
    torch.Tensor
]

class Loss():
    pass

class DissimilarityTransform():
    pass

class DissimilarityMetric():
    pass

# type alias for matrix methods
MatrixMethod = Literal[
    'exact',
    'differentiable',
    'nearest'
]
# TODO: does it make sense to create a DissimilarityMethod
#       class with following subclasses, or should these
#       be subclasses of Dissimilarity?
#       - Exact(...)
#       - Differentiable(...)
#       - Nearest(...)
#       - Precomputed(...)

class Dissimilarity():
    
    def __init__(self,
        metric: Union[Callable, DissimilarityMetric, str] = None,
        method: str = MatrixMethod,
        transform: Union[Callable, DissimilarityTransform] = None,
        verbose: Union[int, bool] = False
        ) -> None:
        
        self.metric = metric
        self.method = method
        self.transform = transform
        self.verbose = verbose

    def matrix(self,
        X: Tensor,
        out_format: Literal['square', 'triagonal'] = 'square',
        n_neighbors: int = None
        ) -> Tensor:

        if self.method == 'exact':
            
            X = _convert_input_to_numpy(X)

            if self.verbose:
                print(f'{datetime.now()}: Calculating pairwise distances')

            distances = pdist(X, metric=self.metric)

            if out_format == 'square':
                matrix = squareform(distances)
            else:
                matrix = distances

        elif self.method == 'differentiable':

            if not isinstance(X, torch.Tensor) or not X.requires_grad:
                warnings.warn(
                    'Method \'differentiable\' operating on tensor \
                        for which no gradients are computed.'
                )
                X = _convert_input_to_torch(X)

            # TODO: add old optional second differentiable method
            #       to account for non-Minkowski metrics
            distances = F.pdist(X)

            if out_format == 'square':
                dim = X.shape[0]
                a, b = torch.triu_indices(dim, dim, offset=1)
                matrix = torch.zeros((dim, dim), device=X.device)
                matrix[[a, b]] = distances
                matrix = matrix + matrix.T
            else:
                matrix = distances


        elif self.method == 'nearest':

            if n_neighbors is None:
                raise Exception('Method \'nearest\' requires value for n_neighbors')

            X = _convert_input_to_numpy(X)

            if self.verbose:
                print(f'{datetime.now()}: Indexing nearest neighbors')

            index = NNDescent(X,
                n_neighbors=n_neighbors,
                metric = self.metric
            )
            neighbors, distances = index.neighbor_graph
            neighbors = neighbors[:, 1:]
            distances = distances[:, 1:]

            row_indices = np.repeat(np.arange(X.shape[0]), n_neighbors-1)
            matrix = csr_matrix((
                distances.ravel(),
                (row_indices, neighbors.ravel())
            ))

        else:
            raise Exception(f'Unknown method {self.method}')

        return matrix

class PrecomputedDissimilarity(Dissimilarity):

    def __init__(self,
        D: Tensor,
        transform: Union[Callable, DissimilarityTransform] = None
        ) -> None:

        super().__init__(transform = transform)

        self.distances = D

    def matrix(self) -> Tensor:
         
         return self.distances




class Dataset(td.Dataset ):
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
        X: Union[np.array, torch.Tensor]
        ) -> torch.Tensor: 
        
        return self.encode(X)

    def encode(self,
        X: Union[np.array, torch.Tensor]
        ) -> torch.Tensor:

        if self.trained:
            return self.encoder(X)
        else:
            raise Exception("Encoder not trained!")

    def train(self):
        pass


def _convert_input_to_numpy(
    X: Tensor) -> np.array:
    
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    elif isinstance(X, base.spmatrix):
        X = X.toarray()
    elif isinstance(X, np.array):
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
    elif isinstance(X, np.array):
        X = torch.tensor(X)
    else:
        raise Exception(f'Input type {type(X)} not supported')

    return X
