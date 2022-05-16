from datetime import datetime
import warnings
from typing import Union, Callable, Literal
import torch
import torch.nn.functional as F
import numpy as np
from scipy.sparse import base
from pynndescent import NNDescent
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform

from .transforms import DissimilarityTransform, Identity, PerplexityBased
from .types import Metric, Tensor
from .utils import report

class Dissimilarity():
    
    def __init__(self,
        metric: Metric = None,
        transform: Union[Callable, DissimilarityTransform] = None
        ) -> None:
        
        self.metric = metric
        self.transform = transform
        self.distances: Tensor = None

    def matrix(self,
        X: Tensor = None,
        out_format: Literal['square', 'triangular'] = None,
        **kwargs) -> Tensor:

        raise NotImplementedError

    def transformed_matrix(self,
        X: Tensor = None,
        out_format: Literal['square', 'triangular'] = None,
        **kwargs) -> Tensor:

        if self.transform is None:
            self.transform = Identity

        return self.transform(self.matrix(
            X,
            out_format,
            **kwargs
        ))

    def _check_input(self,
        X: Tensor = None,
        precomputed = False
        ) -> None:

        if not precomputed:
            if X is None:
                raise Exception(
                    'Missing input data for non-precomputed dissimilarity.'
                )
        elif X is not None:
            try:
                squareform(X)
            except:
                if not len(X.shape) == 2 and X.shape[0] == X.shape[1]:
                    raise ValueError(
                        'Expecting either a square matrix or a vector \
                         representing an upper triangular matrix.'
                        )


class Precomputed(Dissimilarity):

    def __init__(self,
        X: Tensor,
        transform: Union[Callable, DissimilarityTransform] = None
        ) -> None:

        super().__init__(
            transform = transform
        )

        self.distances = X

    def matrix(self,
        X: Tensor = None,
        out_format: Literal['square', 'triangular'] = None,
        **kwargs
        ) -> Tensor:

        self._check_input(X, precomputed=True)
        
        X = self.distances

        if len(X.shape) == 2 and X.shape[0] == X.shape[1]:
            if out_format == 'square':
                matrix = self.distances
            else:
                matrix = self.distances[np.triu_indices(X.shape[0], k=1)]
        else:
            if out_format == 'square':
                matrix = squareform(self.distances)
            else:
                matrix = self.distances
         
        return matrix


class Exact(Dissimilarity):
 
    def __init__(self,
        metric: Metric = None,
        keep_result = True,
        transform: Union[Callable, DissimilarityTransform] = None,
        verbose: Union[int, bool] = False
        ) -> None:

        if metric is None:
            metric = 'euclidean'

        super().__init__(
            metric=metric,
            transform=transform
        )

        self.keep_result = keep_result
        self.verbose = verbose

    def matrix(self,
        X: Tensor = None,
        out_format: Literal['square', 'triangular'] = None,
        **kwargs
        ) -> Tensor:

        self._check_input(X)

        X = _convert_input_to_numpy(X)

        if self.distances is None or not self.keep_result:
            if self.verbose:
                report('Calculating pairwise distances.')
            self.distances = pdist(X, metric=self.metric)
        elif self.verbose:
            report('Using previously calculated distances.')

        if out_format == 'square':
            matrix = squareform(self.distances)
        else:
            matrix = self.distances

        return matrix


class NeighborBased(Dissimilarity):

    def __init__(self,
        n_neighbors: int = None,
        metric: Metric = None,
        transform: Union[Callable, DissimilarityTransform] = None,
        verbose: Union[bool, int] = False
        ) -> None:

        super().__init__(
            transform=transform
        )

        self.n_neighbors = n_neighbors
        self.verbose = verbose
        self.metric = metric
    
    def matrix(self,
        X: Tensor = None,
        out_format: Literal['square', 'triangular'] = None,
        **kwargs
        ) -> Tensor:

        self._check_input(X)

        X = _convert_input_to_numpy(X)

        num_pts = X.shape[0]

        if self.n_neighbors is None:
            if isinstance(self.transform, PerplexityBased):
                self.n_neighbors = min(
                    num_pts - 1,
                    int(3 * self.transform.perplexity)
                )
            else:
                self.n_neighbors = int(0.1 * num_pts)
        else:
            if isinstance(self.transform, PerplexityBased):
                if self.n_neighbors < 3 * self.transform.perplexity:
                    warnings.warn(
                        f'Number of neighbors {self.n_neighbors} ' +
                        'smaller than three times perplexity' +
                        f'{self.transform.perplexity} of transform.'
                    )
        
        if self.verbose:
            report('Indexing nearest neighbors.')

        if self.metric is None:
            self.metric = 'euclidean'
        
        index = NNDescent(X,
            n_neighbors=self.n_neighbors,
            metric = self.metric
        )
        neighbors, distances = index.neighbor_graph
        neighbors = neighbors[:, 1:]
        self.distances = distances[:, 1:]

        row_indices = np.repeat(np.arange(X.shape[0]), self.n_neighbors-1)
        matrix = csr_matrix((
            self.distances.ravel(),
            (row_indices, neighbors.ravel())
        ))

        if out_format == 'square':
            matrix = matrix.toarray()
        elif out_format == 'triangular':
            matrix = matrix[np.triu_indices(X.shape[0], k=1)].A1

        return matrix


class Differentiable(Dissimilarity):

    def __init__(self,
        p: Union[int, float] = 2,
        transform: Union[Callable, DissimilarityTransform] = None,
        verbose: Union[int, bool] = False
        ) -> None:

        super().__init__(
            transform=transform
        )

        self.metric_p = p
        self.verbose = verbose

    def matrix(self,
        X: Tensor = None,
        out_format: Literal['square', 'triangular'] = None,
        **kwargs
        ) -> Tensor:

        self._check_input(X)

        if not isinstance(X, torch.Tensor) or not X.requires_grad:
            warnings.warn(
                'Differentiable dissimilarity operating on tensor ' +
                'for which no gradients are computed.'
            )

        X = _convert_input_to_torch(X)

        # TODO: add old optional second differentiable method
        #       to account for non-Minkowski metrics
        self.distances = F.pdist(X, p=self.metric_p)

        if out_format == 'square':
            dim = X.shape[0]
            a, b = torch.triu_indices(dim, dim, offset=1)
            matrix = torch.zeros((dim, dim), device=X.device)
            matrix[[a, b]] = self.distances
            matrix = matrix + matrix.T
        else:
            matrix = self.distances

        return matrix


def _convert_input_to_numpy(
    X: Tensor) -> np.ndarray:
    
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    elif isinstance(X, base.spmatrix):
        X = X.toarray()
    elif isinstance(X, np.ndarray):
        pass
    else:
        raise TypeError(f'Input type {type(X)} not supported')

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
        raise TypeError(f'Input type {type(X)} not supported')

    return X
