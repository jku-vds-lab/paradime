from datetime import datetime
import warnings
from typing import Union, Callable, Literal, Tuple, Any
import torch
import torch.nn.functional as F
import numpy as np
from scipy.sparse import base
from pynndescent import NNDescent
from scipy.sparse import csr_matrix, base
from scipy.spatial.distance import pdist, squareform
from nptyping import NDArray, Shape, assert_isinstance

from .transforms import DissimilarityTransform, Identity, PerplexityBased
from .types import Metric, Tensor, Diss
from .utils import report


class DissimilarityData():

    def __init__(self,
        diss: Diss) -> None:

        if _is_square_array(diss):
            self = SquareDissimilarityArray(diss) # type:ignore
        elif _is_square_tensor(diss):
            self = SquareDissimilarityTensor(diss) # type:ignore
        elif _is_square_sparse(diss):
            self = SparseDissimilarityArray(diss) # type:ignore
        elif _is_triangular_array(diss):
            self = TriangularDissimilarityArray(diss) # type:ignore
        elif _is_triangular_tensor(diss):
            self = TriangularDissimilarityTensor(diss) # type:ignore
        elif _is_array_tuple(diss):
            self = DissimilarityTuple(diss) # type:ignore
        else:
            raise TypeError(
                f'Input type not supported by {type(self).__name__}.')

    def to_square_array(self) -> 'SquareDissimilarityArray':
        raise NotImplementedError()

    def to_square_tensor(self) -> 'SquareDissimilarityTensor':
        raise NotImplementedError()

    def to_triangular_array(self) -> 'TriangularDissimilarityArray':
        raise NotImplementedError()

    def to_triangular_tensor(self) -> 'TriangularDissimilarityTensor':
        raise NotImplementedError()
    
    def to_array_tuple(self) -> 'DissimilarityTuple':
        raise NotImplementedError()


class SquareDissimilarityArray(DissimilarityData):

    def __init__(self,
        diss: NDArray[Shape['Dim, Dim'], Any]
        ) -> None:

        if not _is_square_array(diss):
            raise ValueError('Expected square array.')

        self.diss = diss

    def to_square_array(self) -> 'SquareDissimilarityArray':
        return self

    def to_square_tensor(self) -> 'SquareDissimilarityTensor':
        return SquareDissimilarityTensor(torch.tensor(self.diss))

    def to_triangular_array(self) -> 'TriangularDissimilarityArray':
        return TriangularDissimilarityArray(squareform(self.diss))

    def to_triangular_tensor(self) -> 'TriangularDissimilarityTensor':
        return TriangularDissimilarityTensor(
            torch.tensor(squareform(self.diss))
        )

    def to_sparse_array(self) -> 'SparseDissimilarityArray':
        return SparseDissimilarityArray(
            csr_matrix(self.diss)
        )

    def to_array_tuple(self) -> 'DissimilarityTuple':
        # get indices of off-diagonal elements
        ones = np.ones((10, 10), dtype=np.int32)
        np.fill_diagonal(ones, 0)
        i, j = np.where(ones)
        diss = self.diss[np.stack((i, j)).T]
        return DissimilarityTuple((diss, j))


class SquareDissimilarityTensor(DissimilarityData):

    def __init__(self,
        diss: torch.Tensor
        ) -> None:

        if not _is_square_tensor(diss):
            raise ValueError('Expected square tensor.')

        self.diss = diss

    def to_square_array(self) -> 'SquareDissimilarityArray':
        return SquareDissimilarityArray(self.diss.detach().numpy())

    def to_square_tensor(self) -> 'SquareDissimilarityTensor':
        return self

    def to_triangular_array(self) -> 'TriangularDissimilarityArray':
        return TriangularDissimilarityArray(
            squareform(self.diss.numpy())
        )

    def to_triangular_tensor(self) -> 'TriangularDissimilarityTensor':
        indices = torch.triu_indices(
            self.diss.shape[0],
            self.diss.shape[1],
            offset=1)
        return TriangularDissimilarityTensor(
            self.diss[indices]
        )

    def to_sparse_array(self) -> 'SparseDissimilarityArray':
        return SparseDissimilarityArray(
            csr_matrix(self.diss.numpy())
        )

    def to_array_tuple(self) -> 'DissimilarityTuple':
        # get indices of off-diagonal elements
        ones = np.ones((10, 10), dtype=np.int32)
        np.fill_diagonal(ones, 0)
        i, j = np.where(ones)
        diss = self.diss.numpy()[np.stack((i, j)).T]
        return DissimilarityTuple((diss, j))


class TriangularDissimilarityArray(DissimilarityData):

    def __init__(self,
        diss: NDArray[Shape['*'], Any]
        ) -> None:

        if not _is_triangular_array(diss):
            raise ValueError(
                'Expected vector-form dissimilarity array.'
            )

        self.diss = diss

    def to_square_array(self) -> 'SquareDissimilarityArray':
        return SquareDissimilarityArray(
            squareform(self.diss)
        )

    def to_square_tensor(self) -> 'SquareDissimilarityTensor':
        return SquareDissimilarityTensor(
            torch.tensor(squareform(self.diss))
        )

    def to_triangular_array(self) -> 'TriangularDissimilarityArray':
        return self

    def to_triangular_tensor(self) -> 'TriangularDissimilarityTensor':
        return TriangularDissimilarityTensor(
            torch.tensor(self.diss)
        )

    def to_sparse_array(self) -> 'SparseDissimilarityArray':
        return SparseDissimilarityArray(
            csr_matrix(squareform(self.diss))
        )

    def to_array_tuple(self) -> 'DissimilarityTuple':
        return self.to_square_array().to_array_tuple()


class TriangularDissimilarityTensor(DissimilarityData):

    def __init__(self,
        diss: torch.Tensor
        ) -> None:

        if not _is_triangular_tensor(diss):
            raise ValueError(
                'Expected vector-form dissimilarity tensor.'
            )

        self.diss = diss

    def to_square_array(self) -> 'SquareDissimilarityArray':
        return SquareDissimilarityArray(
            squareform(self.diss.detach().numpy())
        )

    def to_square_tensor(self) -> 'SquareDissimilarityTensor':
        # get dimensions of square matrix
        d = int(np.ceil(np.sqrt(len(self.diss) * 2)))
        matrix = torch.zeros((d, d), device=self.diss.device)
        a, b = torch.triu_indices(d, d, offset=1)
        matrix[[a, b]] = self.diss
        matrix = matrix + matrix.T
        return SquareDissimilarityTensor(
            matrix
        )

    def to_triangular_array(self) -> 'TriangularDissimilarityArray':
        return self.diss.detach().numpy()

    def to_triangular_tensor(self) -> 'TriangularDissimilarityTensor':
        return self

    def to_sparse_array(self) -> 'SparseDissimilarityArray':
        return SparseDissimilarityArray(
            csr_matrix(squareform(self.to_square_array().to_array_tuple()))
        )

    def to_array_tuple(self) -> 'DissimilarityTuple':
        return self.to_square_array().to_array_tuple()


class SparseDissimilarityArray(DissimilarityData):
    def __init__(self,
        diss: base.spmatrix
        ) -> None:

        if not _is_square_sparse(diss):
            raise ValueError(
                'Expected vector-form dissimilarity tensor.'
            )

        self.diss = diss

    def to_square_array(self) -> 'SquareDissimilarityArray':
        return SquareDissimilarityArray(
            self.diss.toarray()
        )

    def to_square_tensor(self) -> 'SquareDissimilarityTensor':
        return SquareDissimilarityTensor(
            torch.tensor(self.diss.toarray())
        )

    def to_triangular_array(self) -> 'TriangularDissimilarityArray':
        indices = np.triu_indices(self.diss.shape[0], k=1)
        return TriangularDissimilarityArray(
            self.diss[indices].A[0]
        )

    def to_triangular_tensor(self) -> 'TriangularDissimilarityTensor':
        indices = np.triu_indices(self.diss.shape[0], k=1)
        return TriangularDissimilarityTensor(
            torch.tensor(self.diss[indices].A[0])
        )

    def to_sparse_array(self) -> 'SparseDissimilarityArray':
        return self

    def to_array_tuple(self) -> 'DissimilarityTuple':
        return self.to_square_array().to_array_tuple()


class DissimilarityTuple(DissimilarityData):
    def __init__(self,
        diss: Tuple[
            NDArray[Shape['Dim, Nn'], Any],
            NDArray[Shape['Dim, Nn'], Any]
        ]
        ) -> None:

        if not _is_array_tuple(diss):
            raise ValueError(
                'Expected tuple of arrays (neighbors, dissimilarities).'
            )

        self.diss = diss

    def to_square_array(self) -> 'SquareDissimilarityArray':
        return self.to_sparse_array().to_square_array()

    def to_square_tensor(self) -> 'SquareDissimilarityTensor':
        return self.to_sparse_array().to_square_tensor()

    def to_triangular_array(self) -> 'TriangularDissimilarityArray':
        return self.to_sparse_array().to_triangular_array()

    def to_triangular_tensor(self) -> 'TriangularDissimilarityTensor':
        return self.to_sparse_array().to_triangular_tensor()

    def to_sparse_array(self) -> 'SparseDissimilarityArray':
        neighbors, diss = self.diss
        row_indices = np.repeat(
            np.arange(neighbors.shape[0]),
            neighbors.shape[1]
            )
        matrix = csr_matrix((
            diss.ravel(),
            (row_indices, neighbors.ravel())
        ))
        return SparseDissimilarityArray(matrix)

    def to_array_tuple(self) -> 'DissimilarityTuple':
        return self


def _is_square_array(
    diss: Diss
    ) -> bool:

    return assert_isinstance(
        diss,
        NDArray[Shape['Dim, Dim'], Any]
    )

def _is_square_tensor(
    diss: Diss
    ) -> bool:

    result = False

    if isinstance(diss, torch.Tensor):
        if len(diss.shape) == 2 and diss.shape[0] == diss.shape[1]:
            result = True

    return result

def _is_triangular_array(
    diss: Diss
    ) -> bool:

    if assert_isinstance(diss, NDArray[Shape['Dim'], Any]):
        try:
            squareform(diss)
        except:
            return False
        else:
            return True
    else:
            return False

def _is_triangular_tensor(
    diss: Diss
    ) -> bool:

    if isinstance(diss, torch.Tensor):
        try:
            squareform(diss.numpy())
        except:
            return False
        else:
            return True
    else:
        return False

def _is_square_sparse(
    diss: Diss
    ) -> bool:

    result = False

    if isinstance(diss,base.sparse):
        if (len(diss.shape) == 2 and
            diss.shape[0] == diss.shape[1]):
            result = False
    
    return result
        

def _is_array_tuple(
    diss: Diss
    ) -> bool:

    if assert_isinstance(diss,
        Tuple[
            NDArray[Shape['Dim'], Any],
            NDArray[Shape['Dim'], Any]
        ]) and len(diss[0]) == len(diss[1]):
        return True
    else:
        return False


class Dissimilarity():
    
    def __init__(self,
        metric: Metric = None,
        transform: Union[Callable, DissimilarityTransform] = None
        ) -> None:
        
        self.metric = metric
        self.transform = transform

    def compute_dissimilarities(self,
        X: Tensor = None,
        **kwargs) -> DissimilarityData:

        raise NotImplementedError

    # TODO: implement tranforms
    def transformed_dissimilarities(self
        ) -> DissimilarityData:

        raise NotImplementedError()


class Precomputed(Dissimilarity):

    def __init__(self,
        X: Tensor,
        transform: Union[Callable, DissimilarityTransform] = None
        ) -> None:

        super().__init__(
            transform = transform
        )

        self.dissimilarities = DissimilarityData(X)

    def compute_dissimilarities(self,
        X: Tensor = None,
        **kwargs
        ) -> DissimilarityData:

        if X is not None:
            warnings.warn('Ignoring input for precomputed dissimilarity')
        
        return self.dissimilarities


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

    def compute_dissimilarities(self,
        X: Tensor = None,
        **kwargs
        ) -> DissimilarityData:

        if X is None:
            raise ValueError(
                'Missing input for non-precomputed dissimilarity.'
            )

        X = _convert_input_to_numpy(X)

        if hasattr(self, 'dissimilarities') or not self.keep_result:
            if self.verbose:
                report('Calculating pairwise distances.')
            self.dissimilarities = DissimilarityData(
                pdist(X, metric=self.metric)
            )
        elif self.verbose:
            report('Using previously calculated distances.')

        return self.dissimilarities


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
    
    def compute_dissimilarities(self,
        X: Tensor = None,
        **kwargs
        ) -> DissimilarityData:

        if X is None:
            raise ValueError(
                'Missing input for non-precomputed dissimilarity.'
            )

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

        self.dissimilarities = DissimilarityData(
            (neighbors, distances)
        )

        return self.dissimilarities


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
        **kwargs
        ) -> Tensor:

        if X is None:
            raise ValueError(
                'Missing input for non-precomputed dissimilarity.'
            )

        if not isinstance(X, torch.Tensor) or not X.requires_grad:
            warnings.warn(
                'Differentiable dissimilarity operating on tensor ' +
                'for which no gradients are computed.'
            )

        X = _convert_input_to_torch(X)

        # TODO: add old optional second differentiable method
        #       to account for non-Minkowski metrics
        self.dissimilarities = TriangularDissimilarityTensor(
            F.pdist(X, p=self.metric_p)
        )

        return self.dissimilarities



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
