from typing import Tuple, Any
import torch
import numpy as np
import scipy.sparse
from scipy.spatial.distance import squareform

from .types import Metric, Tensor, Diss
from .utils import report

class AffinityData():

    def __init__(self):
        self.diss = None

    def to_square_array(self) -> 'SquareAffinityArray':
        raise NotImplementedError()

    def to_square_tensor(self) -> 'SquareAffinityTensor':
        raise NotImplementedError()

    def to_triangular_array(self) -> 'TriangularAffinityArray':
        raise NotImplementedError()

    def to_triangular_tensor(self) -> 'TriangularAffinityTensor':
        raise NotImplementedError()
    
    def to_array_tuple(self) -> 'AffinityTuple':
        raise NotImplementedError()


def affinity_factory(
    diss: Diss) -> AffinityData:

    if _is_square_array(diss):
        dd = SquareAffinityArray(diss) # type:ignore
    elif _is_square_tensor(diss):
        dd = SquareAffinityTensor(diss) # type:ignore
    elif _is_square_sparse(diss):
        dd = SparseAffinityArray(diss) # type:ignore
    elif _is_triangular_array(diss):
        dd = TriangularAffinityArray(diss) # type:ignore
    elif _is_triangular_tensor(diss):
        dd = TriangularAffinityTensor(diss) # type:ignore
    elif _is_array_tuple(diss):
        dd = AffinityTuple(diss) # type:ignore
    else:
        raise TypeError(
            f'Input type not supported by {AffinityData.__name__}.')

    return dd


class SquareAffinityArray(AffinityData):

    def __init__(self,
        diss: np.ndarray
        ) -> None:

        if not _is_square_array(diss):
            raise ValueError('Expected square array.')

        self.diss = diss

    def to_square_array(self) -> 'SquareAffinityArray':
        return self

    def to_square_tensor(self) -> 'SquareAffinityTensor':
        return SquareAffinityTensor(torch.tensor(self.diss))

    def to_triangular_array(self) -> 'TriangularAffinityArray':
        return TriangularAffinityArray(squareform(self.diss))

    def to_triangular_tensor(self) -> 'TriangularAffinityTensor':
        return TriangularAffinityTensor(
            torch.tensor(squareform(self.diss))
        )

    def to_sparse_array(self) -> 'SparseAffinityArray':
        return SparseAffinityArray(
            scipy.sparse.csr_matrix(self.diss)
        )

    def to_array_tuple(self) -> 'AffinityTuple':
        # # get indices of off-diagonal elements
        # ones = np.ones(self.diss.shape, dtype=np.int32)
        # np.fill_diagonal(ones, 0)
        # i, j = np.where(ones)
        # return AffinityTuple((
        #     self.diss[i,j].reshape(-1, self.diss.shape[0] - 1),
        #     j.reshape(-1, self.diss.shape[0] - 1)
        # ))
        return AffinityTuple((
            self.diss.reshape(-1, self.diss.shape[0]),
            np.tile(
                np.arange(self.diss.shape[0]),
                self.diss.shape[0])
        ))


class SquareAffinityTensor(AffinityData):

    def __init__(self,
        diss: torch.Tensor
        ) -> None:

        if not _is_square_tensor(diss):
            raise ValueError('Expected square tensor.')

        self.diss = diss

    def to_square_array(self) -> 'SquareAffinityArray':
        return SquareAffinityArray(self.diss.detach().numpy())

    def to_square_tensor(self) -> 'SquareAffinityTensor':
        return self

    def to_triangular_array(self) -> 'TriangularAffinityArray':
        return TriangularAffinityArray(
            squareform(self.diss.detach().numpy())
        )

    def to_triangular_tensor(self) -> 'TriangularAffinityTensor':
        i, j = torch.triu_indices(
            self.diss.shape[0],
            self.diss.shape[1],
            offset=1)
        return TriangularAffinityTensor(
            self.diss[i,j]
        )

    def to_sparse_array(self) -> 'SparseAffinityArray':
        return SparseAffinityArray(
            scipy.sparse.csr_matrix(self.diss.detach().numpy())
        )

    def to_array_tuple(self) -> 'AffinityTuple':
        diss = self.diss.detach().numpy()
        # get indices of off-diagonal elements
        ones = np.ones(diss.shape, dtype=np.int32)
        np.fill_diagonal(ones, 0)
        i, j = np.where(ones)
        return AffinityTuple((
            diss[i,j].reshape(-1, self.diss.shape[0] - 1),
            j.reshape(-1, self.diss.shape[0] - 1)
        ))


class TriangularAffinityArray(AffinityData):

    def __init__(self,
        diss: np.ndarray
        ) -> None:

        if not _is_triangular_array(diss):
            raise ValueError(
                'Expected vector-form affinity array.'
            )

        self.diss = diss

    def to_square_array(self) -> 'SquareAffinityArray':
        return SquareAffinityArray(
            squareform(self.diss)
        )

    def to_square_tensor(self) -> 'SquareAffinityTensor':
        return SquareAffinityTensor(
            torch.tensor(squareform(self.diss))
        )

    def to_triangular_array(self) -> 'TriangularAffinityArray':
        return self

    def to_triangular_tensor(self) -> 'TriangularAffinityTensor':
        return TriangularAffinityTensor(
            torch.tensor(self.diss)
        )

    def to_sparse_array(self) -> 'SparseAffinityArray':
        return SparseAffinityArray(
            scipy.sparse.csr_matrix(squareform(self.diss))
        )

    def to_array_tuple(self) -> 'AffinityTuple':
        return self.to_square_array().to_array_tuple()


class TriangularAffinityTensor(AffinityData):

    def __init__(self,
        diss: torch.Tensor
        ) -> None:

        if not _is_triangular_tensor(diss):
            raise ValueError(
                'Expected vector-form affinity tensor.'
            )

        self.diss = diss

    def to_square_array(self) -> 'SquareAffinityArray':
        return SquareAffinityArray(
            squareform(self.diss.detach().numpy())
        )

    def to_square_tensor(self) -> 'SquareAffinityTensor':
        # get dimensions of square matrix
        d = int(np.ceil(np.sqrt(len(self.diss) * 2)))
        matrix = torch.zeros((d, d), device=self.diss.device)
        a, b = torch.triu_indices(d, d, offset=1)
        matrix[[a, b]] = self.diss
        matrix = matrix + matrix.T
        return SquareAffinityTensor(
            matrix
        )

    def to_triangular_array(self) -> 'TriangularAffinityArray':
        return self.diss.detach().numpy()

    def to_triangular_tensor(self) -> 'TriangularAffinityTensor':
        return self

    def to_sparse_array(self) -> 'SparseAffinityArray':
        return SparseAffinityArray(
            scipy.sparse.csr_matrix(squareform(self.to_square_array().to_array_tuple()))
        )

    def to_array_tuple(self) -> 'AffinityTuple':
        return self.to_square_array().to_array_tuple()


class SparseAffinityArray(AffinityData):
    def __init__(self,
        diss: scipy.sparse.spmatrix
        ) -> None:

        if not _is_square_sparse(diss):
            raise ValueError(
                'Expected vector-form affinity tensor.'
            )

        self.diss = diss

    def to_square_array(self) -> 'SquareAffinityArray':
        return SquareAffinityArray(
            self.diss.toarray()
        )

    def to_square_tensor(self) -> 'SquareAffinityTensor':
        return SquareAffinityTensor(
            torch.tensor(self.diss.toarray())
        )

    def to_triangular_array(self) -> 'TriangularAffinityArray':
        indices = np.triu_indices(self.diss.shape[0], k=1)
        return TriangularAffinityArray(
            self.diss[indices].A[0]
        )

    def to_triangular_tensor(self) -> 'TriangularAffinityTensor':
        indices = np.triu_indices(self.diss.shape[0], k=1)
        return TriangularAffinityTensor(
            torch.tensor(self.diss[indices].A[0])
        )

    def to_sparse_array(self) -> 'SparseAffinityArray':
        return self

    def to_array_tuple(self) -> 'AffinityTuple':
        return self.to_square_array().to_array_tuple()


class AffinityTuple(AffinityData):
    def __init__(self,
        diss: Tuple[
            np.ndarray,
            np.ndarray
        ]
        ) -> None:

        if not _is_array_tuple(diss):
            raise ValueError(
                'Expected tuple of arrays (neighbors, affinities).'
            )

        # sort entries by ascending distance
        neighbors, distances = diss
        indices = distances.argsort()
        neighbors = np.take_along_axis(neighbors, indices, axis=1)
        distances = np.take_along_axis(distances, indices, axis=1)

        self.diss = (neighbors, distances)

    def to_square_array(self) -> 'SquareAffinityArray':
        return self.to_sparse_array().to_square_array()

    def to_square_tensor(self) -> 'SquareAffinityTensor':
        return self.to_sparse_array().to_square_tensor()

    def to_triangular_array(self) -> 'TriangularAffinityArray':
        return self.to_sparse_array().to_triangular_array()

    def to_triangular_tensor(self) -> 'TriangularAffinityTensor':
        return self.to_sparse_array().to_triangular_tensor()

    def to_sparse_array(self) -> 'SparseAffinityArray':
        neighbors, diss = self.diss
        row_indices = np.repeat(
            np.arange(neighbors.shape[0]),
            neighbors.shape[1]
            )
        matrix = scipy.sparse.csr_matrix((
            diss.ravel(),
            (row_indices, neighbors.ravel())
        ))
        return SparseAffinityArray(matrix)

    def to_array_tuple(self) -> 'AffinityTuple':
        return self


def _is_square_array(
    diss: Diss
    ) -> bool:

    result = False

    if isinstance(diss, np.ndarray):
        if len(diss.shape) == 2 and diss.shape[0] == diss.shape[1]:
            result = True
    
    return result

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

    result = False

    if isinstance(diss, np.ndarray) and len(diss.shape) == 1:
        try:
            squareform(diss)
        except:
            pass
        else:
            result = True
    
    return result

def _is_triangular_tensor(
    diss: Diss
    ) -> bool:

    result = False

    if isinstance(diss, torch.Tensor):
        try:
            squareform(diss.detach().numpy())
        except:
            pass
        else:
            result = True
    
    return result

def _is_square_sparse(
    diss: Diss
    ) -> bool:

    result = False

    if isinstance(diss, scipy.sparse.spmatrix):
        if (len(diss.shape) == 2 and
            diss.shape[0] == diss.shape[1]):
            result = True
    
    return result

def _is_array_tuple(
    diss: Diss
    ) -> bool:

    result = False

    if isinstance(diss, tuple) and len(diss) == 2:
        if len(diss[0].shape) == 2 and diss[0].shape == diss[1].shape:
            result = True
    
    return result