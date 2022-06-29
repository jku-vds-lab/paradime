from typing import Tuple, Any
import torch
import numpy as np
import scipy.sparse
from scipy.spatial.distance import squareform

from .types import Rels
from .utils import report

class RelationData():

    def __init__(self):
        self.data = None

    def to_square_array(self) -> 'SquareRelationArray':
        raise NotImplementedError()

    def to_square_tensor(self) -> 'SquareRelationTensor':
        raise NotImplementedError()

    def to_triangular_array(self) -> 'TriangularRelationArray':
        raise NotImplementedError()

    def to_triangular_tensor(self) -> 'TriangularRelationTensor':
        raise NotImplementedError()
    
    def to_array_tuple(self) -> 'NeighborRelationTuple':
        raise NotImplementedError()

    def to_sparse_array(self) -> 'SparseRelationArray':
        raise NotImplementedError()


def relation_factory(
    rels: Rels) -> RelationData:

    if _is_square_array(rels):
        dd = SquareRelationArray(rels) # type:ignore
    elif _is_square_tensor(rels):
        dd = SquareRelationTensor(rels) # type:ignore
    elif _is_square_sparse(rels):
        dd = SparseRelationArray(rels) # type:ignore
    elif _is_triangular_array(rels):
        dd = TriangularRelationArray(rels) # type:ignore
    elif _is_triangular_tensor(rels):
        dd = TriangularRelationTensor(rels) # type:ignore
    elif _is_array_tuple(rels):
        dd = NeighborRelationTuple(rels) # type:ignore
    else:
        raise TypeError(
            f'Input type not supported by {RelationData.__name__}.')

    return dd


class SquareRelationArray(RelationData):

    def __init__(self,
        rels: np.ndarray
        ) -> None:

        if not _is_square_array(rels):
            raise ValueError('Expected square array.')

        self.data = rels

    def to_square_array(self) -> 'SquareRelationArray':
        return self

    def to_square_tensor(self) -> 'SquareRelationTensor':
        return SquareRelationTensor(torch.tensor(self.data))

    def to_triangular_array(self) -> 'TriangularRelationArray':
        return TriangularRelationArray(squareform(self.data))

    def to_triangular_tensor(self) -> 'TriangularRelationTensor':
        return TriangularRelationTensor(
            torch.tensor(squareform(self.data))
        )

    def to_sparse_array(self) -> 'SparseRelationArray':
        return SparseRelationArray(
            scipy.sparse.csr_matrix(self.data)
        )

    def to_array_tuple(self) -> 'NeighborRelationTuple':
        # # get indices of off-diagonal elements
        # ones = np.ones(self.data.shape, dtype=np.int32)
        # np.fill_diagonal(ones, 0)
        # i, j = np.where(ones)
        # return AffinityTuple((
        #     self.data[i,j].reshape(-1, self.data.shape[0] - 1),
        #     j.reshape(-1, self.data.shape[0] - 1)
        # ))
        return NeighborRelationTuple((
            np.tile(
                np.arange(self.data.shape[0]),
                (self.data.shape[0], 1)
            ),
            self.data
        ))


class SquareRelationTensor(RelationData):

    def __init__(self,
        rels: torch.Tensor
        ) -> None:

        if not _is_square_tensor(rels):
            raise ValueError('Expected square tensor.')

        self.data = rels

    def to_square_array(self) -> 'SquareRelationArray':
        return SquareRelationArray(self.data.detach().numpy())

    def to_square_tensor(self) -> 'SquareRelationTensor':
        return self

    def to_triangular_array(self) -> 'TriangularRelationArray':
        return TriangularRelationArray(
            squareform(self.data.detach().numpy())
        )

    def to_triangular_tensor(self) -> 'TriangularRelationTensor':
        i, j = torch.triu_indices(
            self.data.shape[0],
            self.data.shape[1],
            offset=1)
        return TriangularRelationTensor(
            self.data[i,j]
        )

    def to_sparse_array(self) -> 'SparseRelationArray':
        return SparseRelationArray(
            scipy.sparse.csr_matrix(self.data.detach().numpy())
        )

    def to_array_tuple(self) -> 'NeighborRelationTuple':
        rels = self.data.detach().numpy()
        return NeighborRelationTuple((
            np.tile(
                np.arange(rels.shape[0]),
                (rels.shape[0], 1)
            ),
            rels
        ))


class TriangularRelationArray(RelationData):

    def __init__(self,
        rels: np.ndarray
        ) -> None:

        if not _is_triangular_array(rels):
            raise ValueError(
                'Expected vector-form relation array.'
            )

        self.data = rels

    def to_square_array(self) -> 'SquareRelationArray':
        return SquareRelationArray(
            squareform(self.data)
        )

    def to_square_tensor(self) -> 'SquareRelationTensor':
        return SquareRelationTensor(
            torch.tensor(squareform(self.data))
        )

    def to_triangular_array(self) -> 'TriangularRelationArray':
        return self

    def to_triangular_tensor(self) -> 'TriangularRelationTensor':
        return TriangularRelationTensor(
            torch.tensor(self.data)
        )

    def to_sparse_array(self) -> 'SparseRelationArray':
        return SparseRelationArray(
            scipy.sparse.csr_matrix(squareform(self.data))
        )

    def to_array_tuple(self) -> 'NeighborRelationTuple':
        return self.to_square_array().to_array_tuple()


class TriangularRelationTensor(RelationData):

    def __init__(self,
        rels: torch.Tensor
        ) -> None:

        if not _is_triangular_tensor(rels):
            raise ValueError(
                'Expected vector-form relation tensor.'
            )

        self.data = rels

    def to_square_array(self) -> 'SquareRelationArray':
        return SquareRelationArray(
            squareform(self.data.detach().numpy())
        )

    def to_square_tensor(self) -> 'SquareRelationTensor':
        # get dimensions of square matrix
        d = int(np.ceil(np.sqrt(len(self.data) * 2)))
        matrix = torch.zeros((d, d),
            dtype=self.data.dtype,
            device=self.data.device
        )
        a, b = torch.triu_indices(d, d, offset=1)
        matrix[[a, b]] = self.data
        matrix = matrix + matrix.T
        return SquareRelationTensor(
            matrix
        )

    def to_triangular_array(self) -> 'TriangularRelationArray':
        return TriangularRelationArray(
            self.data.detach().numpy()
        )

    def to_triangular_tensor(self) -> 'TriangularRelationTensor':
        return self

    def to_sparse_array(self) -> 'SparseRelationArray':
        return self.to_triangular_array().to_sparse_array()

    def to_array_tuple(self) -> 'NeighborRelationTuple':
        return self.to_square_array().to_array_tuple()


class SparseRelationArray(RelationData):
    def __init__(self,
        rels: scipy.sparse.spmatrix
        ) -> None:

        if not _is_square_sparse(rels):
            raise ValueError(
                'Expected vector-form relation tensor.'
            )

        self.data = rels

    def to_square_array(self) -> 'SquareRelationArray':
        return SquareRelationArray(
            self.data.toarray()
        )

    def to_square_tensor(self) -> 'SquareRelationTensor':
        return SquareRelationTensor(
            torch.tensor(self.data.toarray())
        )

    def to_triangular_array(self) -> 'TriangularRelationArray':
        indices = np.triu_indices(self.data.shape[0], k=1)
        return TriangularRelationArray(
            self.data[indices].A[0]
        )

    def to_triangular_tensor(self) -> 'TriangularRelationTensor':
        indices = np.triu_indices(self.data.shape[0], k=1)
        return TriangularRelationTensor(
            torch.tensor(self.data[indices].A[0])
        )

    def to_sparse_array(self) -> 'SparseRelationArray':
        return self

    def to_array_tuple(self) -> 'NeighborRelationTuple':
        return self.to_square_array().to_array_tuple()


class NeighborRelationTuple(RelationData):
    def __init__(self,
        reldata: Tuple[
            np.ndarray,
            np.ndarray
        ]
        ) -> None:

        if not _is_array_tuple(reldata):
            raise ValueError(
                'Expected tuple of arrays (neighbors, relations).'
            )

        # sort entries by ascending relation values
        neighbors, rels = reldata
        indices = rels.argsort()
        neighbors = np.take_along_axis(neighbors, indices, axis=1)
        rels = np.take_along_axis(rels, indices, axis=1)

        self.data = (neighbors, rels)

    def to_square_array(self) -> 'SquareRelationArray':
        return self.to_sparse_array().to_square_array()

    def to_square_tensor(self) -> 'SquareRelationTensor':
        return self.to_sparse_array().to_square_tensor()

    def to_triangular_array(self) -> 'TriangularRelationArray':
        return self.to_sparse_array().to_triangular_array()

    def to_triangular_tensor(self) -> 'TriangularRelationTensor':
        return self.to_sparse_array().to_triangular_tensor()

    def to_sparse_array(self) -> 'SparseRelationArray':
        neighbors, rels = self.data
        row_indices = np.repeat(
            np.arange(neighbors.shape[0]),
            neighbors.shape[1]
            )
        matrix = scipy.sparse.csr_matrix((
            rels.ravel(),
            (row_indices, neighbors.ravel())
        ))
        return SparseRelationArray(matrix)

    def to_array_tuple(self) -> 'NeighborRelationTuple':
        return self


def _is_square_array(
    rels: Rels
    ) -> bool:

    result = False

    if isinstance(rels, np.ndarray):
        if len(rels.shape) == 2 and rels.shape[0] == rels.shape[1]:
            result = True
    
    return result

def _is_square_tensor(
    rels: Rels
    ) -> bool:

    result = False

    if isinstance(rels, torch.Tensor):
        if len(rels.shape) == 2 and rels.shape[0] == rels.shape[1]:
            result = True

    return result

def _is_triangular_array(
    rels: Rels
    ) -> bool:

    result = False

    if isinstance(rels, np.ndarray) and len(rels.shape) == 1:
        try:
            squareform(rels)
        except:
            pass
        else:
            result = True
    
    return result

def _is_triangular_tensor(
    rels: Rels
    ) -> bool:

    result = False

    if isinstance(rels, torch.Tensor):
        try:
            squareform(rels.detach().numpy())
        except:
            pass
        else:
            result = True
    
    return result

def _is_square_sparse(
    rels: Rels
    ) -> bool:

    result = False

    if isinstance(rels, scipy.sparse.spmatrix):
        if (len(rels.shape) == 2 and
            rels.shape[0] == rels.shape[1]):
            result = True
    
    return result

def _is_array_tuple(
    rels: Rels
    ) -> bool:

    result = False

    if isinstance(rels, tuple) and len(rels) == 2:
        if len(rels[0].shape) == 2 and rels[0].shape == rels[1].shape:
            result = True
    
    return result