from typing import Tuple, Any
import torch
import numpy as np
import scipy.sparse
from scipy.spatial.distance import squareform
from nptyping import NDArray, Shape, assert_isinstance

from .types import Metric, Tensor, Diss
from .utils import report

class DissimilarityData():

    def __init__(self):
        self.diss = None

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


def dissimilarity_factory(
    diss: Diss) -> DissimilarityData:

    if _is_square_array(diss):
        dd = SquareDissimilarityArray(diss) # type:ignore
    elif _is_square_tensor(diss):
        dd = SquareDissimilarityTensor(diss) # type:ignore
    elif _is_square_sparse(diss):
        dd = SparseDissimilarityArray(diss) # type:ignore
    elif _is_triangular_array(diss):
        dd = TriangularDissimilarityArray(diss) # type:ignore
    elif _is_triangular_tensor(diss):
        dd = TriangularDissimilarityTensor(diss) # type:ignore
    elif _is_array_tuple(diss):
        dd = DissimilarityTuple(diss) # type:ignore
    else:
        raise TypeError(
            f'Input type not supported by {DissimilarityData.__name__}.')

    return dd


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
            scipy.sparse.csr_matrix(self.diss)
        )

    def to_array_tuple(self) -> 'DissimilarityTuple':
        # # get indices of off-diagonal elements
        # ones = np.ones(self.diss.shape, dtype=np.int32)
        # np.fill_diagonal(ones, 0)
        # i, j = np.where(ones)
        # return DissimilarityTuple((
        #     self.diss[i,j].reshape(-1, self.diss.shape[0] - 1),
        #     j.reshape(-1, self.diss.shape[0] - 1)
        # ))
        return DissimilarityTuple((
            self.diss.reshape(-1, self.diss.shape[0]),
            np.tile(
                np.arange(self.diss.shape[0]),
                self.diss.shape[0])
        ))


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
            squareform(self.diss.detach().numpy())
        )

    def to_triangular_tensor(self) -> 'TriangularDissimilarityTensor':
        i, j = torch.triu_indices(
            self.diss.shape[0],
            self.diss.shape[1],
            offset=1)
        return TriangularDissimilarityTensor(
            self.diss[i,j]
        )

    def to_sparse_array(self) -> 'SparseDissimilarityArray':
        return SparseDissimilarityArray(
            scipy.sparse.csr_matrix(self.diss.detach().numpy())
        )

    def to_array_tuple(self) -> 'DissimilarityTuple':
        diss = self.diss.detach().numpy()
        # get indices of off-diagonal elements
        ones = np.ones(diss.shape, dtype=np.int32)
        np.fill_diagonal(ones, 0)
        i, j = np.where(ones)
        return DissimilarityTuple((
            diss[i,j].reshape(-1, self.diss.shape[0] - 1),
            j.reshape(-1, self.diss.shape[0] - 1)
        ))


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
            scipy.sparse.csr_matrix(squareform(self.diss))
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
            scipy.sparse.csr_matrix(squareform(self.to_square_array().to_array_tuple()))
        )

    def to_array_tuple(self) -> 'DissimilarityTuple':
        return self.to_square_array().to_array_tuple()


class SparseDissimilarityArray(DissimilarityData):
    def __init__(self,
        diss: scipy.sparse.spmatrix
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

        # sort entries by ascending distance
        neighbors, distances = diss
        indices = distances.argsort()
        neighbors = neighbors[indices]
        distances = distances[indices]

        self.diss = (neighbors, distances)

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
        matrix = scipy.sparse.csr_matrix((
            diss.ravel(),
            (row_indices, neighbors.ravel())
        ))
        return SparseDissimilarityArray(matrix)

    def to_array_tuple(self) -> 'DissimilarityTuple':
        return self


def _is_square_array(
    diss: Diss
    ) -> bool:

    result = False

    if isinstance(diss, NDArray):
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

    if isinstance(diss, NDArray) and len(diss.shape) == 1:
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