from typing import Tuple, Any
import functools
import itertools
import torch
import numpy as np
import scipy.sparse
from scipy.spatial.distance import squareform

from .types import IndexList, Rels
from .utils import report

class RelationData():
    """Base class for storing relations between data points."""

    def __init__(self):
        self.data = None

    def sub(self, indices: IndexList) -> torch.Tensor:
        """Subsamples the relation matrix based on item indices.
        
        Args:
            indices: A flat list of item indices.
            
        Returns:
            A square PyTorch tensor consisting of all relations
            between items with the given indices, intended to be
            used for batch-wise subsampling of global relations.
        """
        raise NotImplementedError(
            f"Matrix subsampling not implemented for {type(self).__name__}."
        )

    def to_flat_array(self) -> 'FlatRelationArray':
        """Converts the relations to a :class:`FlatRelationArray`.
        
        Returns:
            The converted relations.
        """
        raise NotImplementedError(
            "Conversion to flat array not "
            f"implemented for {type(self).__name__}."
        )

    def to_flat_tensor(self) -> 'FlatRelationTensor':
        """Converts the relations to a :class:`FlatRelationTensor`.
        
        Returns:
            The converted relations.
        """
        raise NotImplementedError(
            "Conversion to flat tensor not "
            f"implemented for {type(self).__name__}."
        )

    def to_square_array(self) -> 'SquareRelationArray':
        """Converts the relations to a :class:`SquareRelationArray`.
        
        Returns:
            The converted relations.
        """
        raise NotImplementedError(
            "Conversion to square array not "
            f"implemented for {type(self).__name__}."
        )

    def to_square_tensor(self) -> 'SquareRelationTensor':
        """Converts the relations to a :class:`SquareRelationTensor`.
        
        Returns:
            The converted relations.
        """
        raise NotImplementedError(
            "Conversion to square tensor not "
            f"implemented for {type(self).__name__}."
        )

    def to_triangular_array(self) -> 'TriangularRelationArray':
        """Converts the relations to a :class:`TriangularRelationArray`.
        
        Returns:
            The converted relations.
        """
        raise NotImplementedError(
            "Conversion to triangular array not "
            f"implemented for {type(self).__name__}."
        )

    def to_triangular_tensor(self) -> 'TriangularRelationTensor':
        """Converts the relations to a :class:`TriangularRelationTensor`.
        
        Returns:
            The converted relations.
        """
        raise NotImplementedError(
            "Conversion to triangular tensor not "
            f"implemented for {type(self).__name__}."
        )
    
    def to_array_tuple(self) -> 'NeighborRelationTuple':
        """Converts the relations to a :class:`NeighborRelationTuple`.
        
        Returns:
            The converted relations.
        """
        raise NotImplementedError(
            "Conversion to neighbor relation tuple not "
            f"implemented for {type(self).__name__}."
        )

    def to_sparse_array(self) -> 'SparseRelationArray':
        """Converts the relations to a :class:`SparseRelationArray`.
        
        Returns:
            The converted relations.
        """
        raise NotImplementedError(
            "Conversion to sparse array not "
            f"implemented for {type(self).__name__}."
        )


def relation_factory(
    relations: Rels,
    force_flat: bool = False
) -> RelationData:
    """Creates a :class:`RelationData` object from a variety of input formats.
    
    Args:
        relations: The relations, specified either as a flat array or tensor,
            a square array or tensor, a vector-form (triangular) array or
            tensor, a sparse array, or a tuple (n, r), where n is an array of
            neighor indices for each data point and r is an array of relation
            values of the same shape.
        force_flat: If set true, disables the check for triangular arrays and
            tensors. Useful if flat relation data might have a length equal
            to a triangular number.
    
    Returns:
        A :class:`RelationData` object with a subclass depending on the
        input format.
    """

    if _is_square_array(relations):
        rd = SquareRelationArray(relations)  # type:ignore
    elif _is_square_tensor(relations):
        rd = SquareRelationTensor(relations)  # type:ignore
    elif _is_square_sparse(relations):
        rd = SparseRelationArray(relations)  # type:ignore
    elif _is_triangular_array(relations) and not force_flat:
        rd = TriangularRelationArray(relations)  # type:ignore
    elif _is_triangular_tensor(relations) and not force_flat:
        rd = TriangularRelationTensor(relations)  # type:ignore
    elif _is_flat_array(relations):
        rd = FlatRelationArray(relations)  # type:ignore
    elif _is_flat_tensor(relations):
        rd = FlatRelationTensor(relations)  # type:ignore
    elif _is_array_tuple(relations):
        rd = NeighborRelationTuple(relations)  # type:ignore
    else:
        raise TypeError(
            f'Input type not supported by {RelationData.__name__}.')

    return rd

class FlatRelationArray(RelationData):
    """Relation data in the form of a flat array of individual relations.
    
    Args:
        relations: A flat numpy array of relation values.

    Attributes:
        data: The raw relation data.
    """

    def __init__(self,
        relations: np.ndarray
        ) -> None:

        if not _is_flat_array(relations):
            raise ValueError('Expected vector-form array.')

        self.data = relations

    def to_flat_array(self) -> 'FlatRelationArray':
        return self

    def to_flat_tensor(self) -> 'FlatRelationTensor':
        return FlatRelationTensor(torch.tensor(self.data))

class FlatRelationTensor(RelationData):
    """Relation data in the form of a flat tensor of individual relations.
    
    Args:
        relations: A flat PyTorch tensor of relation values.

    Attributes:
        data: The raw relation data.
    """

    def __init__(self, relations: torch.Tensor) -> None:

        if not _is_flat_array(relations):
            raise ValueError('Expected vector-form tensor.')

        self.data = relations

    def to_flat_array(self) -> 'FlatRelationArray':
        return FlatRelationArray(self.data.detach().numpy())

    def to_flat_tensor(self) -> 'FlatRelationTensor':
        return self

class SquareRelationArray(RelationData):
    """Relation data in the form of a square array.
    
    Args:
        relations: A square numpy array of relation values.

    Attributes:
        data: The raw relation data.
    """

    def __init__(self,relations: np.ndarray) -> None:

        if not _is_square_array(relations):
            raise ValueError('Expected square array.')

        self.data = relations

    def sub(self, indices: IndexList) -> torch.Tensor:
        dim = len(indices)
        indices = np.array(np.meshgrid(indices, indices)).T.reshape(-1,2).T
        return torch.tensor(
            np.reshape(self.data[indices[0], indices[1]], (dim, dim))
        )

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
        return NeighborRelationTuple((
            np.tile(
                np.arange(self.data.shape[0]),
                (self.data.shape[0], 1)
            ),
            self.data
        ))


class SquareRelationTensor(RelationData):
    """Relation data in the form of a square tensor.
    
    Args:
        relations: A square PyTorch tensor of relation values.

    Attributes:
        data: The raw relation data.
    """

    def __init__(self, relations: torch.Tensor) -> None:

        if not _is_square_tensor(relations):
            raise ValueError('Expected square tensor.')

        self.data = relations

    def sub(self, indices: IndexList) -> torch.Tensor:
        dim = len(indices)
        indices = np.array(np.meshgrid(indices, indices)).T.reshape(-1,2).T
        return self.data[indices[0], indices[1]].reshape(dim, dim)

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
    """Relation data in 'triangular' vector-form.
    
    Args:
        relations: A numpy array of relation values, as accepted by
        :func:`scipy.spatial.distance.squareform`.

    Attributes:
        data: The raw relation data.
    """

    def __init__(self, relations: np.ndarray) -> None:

        if not _is_triangular_array(relations):
            raise ValueError(
                'Expected vector-form relation array.'
            )

        self.data = relations

    def sub(self, indices: IndexList) -> torch.Tensor:
        dim = _get_orig_dim(len(self.data))
        combos = itertools.combinations(indices, 2)
        return torch.tensor(squareform(np.array(
            [ self.data[_rowcol_to_triu_index(i, j, dim)] for i, j in combos ]
        )))

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
    """Relation data in 'triangular' vector-form.
    
    Args:
        relations: A PyTorch tensor of relation values, with a
        shape as accepted by :func:`scipy.spatial.distance.squareform`.

    Attributes:
        data: The raw relation data.
    """

    def __init__(self, relations: torch.Tensor) -> None:

        if not _is_triangular_tensor(relations):
            raise ValueError(
                'Expected vector-form relation tensor.'
            )

        self.data = relations

    def sub(self, indices: IndexList) -> torch.Tensor:
        dim = _get_orig_dim(len(self.data))
        combos = itertools.combinations(indices, 2)
        return _tensor_squareform(torch.tensor(
            [ self.data[_rowcol_to_triu_index(i, j, dim)] for i, j in combos ]
        ))

    def to_square_array(self) -> 'SquareRelationArray':
        return SquareRelationArray(
            squareform(self.data.detach().numpy())
        )

    def to_square_tensor(self) -> 'SquareRelationTensor':
        # get dimensions of square matrix
        d = _get_orig_dim(len(self.data))
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
    """Relation data in sparse array form.
    
    Args:
        relations: A square, sparse scipy array of relation values.

    Attributes:
        data: The raw relation data.
    """

    def __init__(self, rels: scipy.sparse.spmatrix) -> None:

        if not _is_square_sparse(rels):
            raise ValueError(
                'Expected vector-form relation tensor.'
            )

        self.data = rels

    def sub(self, indices: IndexList) -> torch.Tensor:
        dim = len(indices)
        indices = np.array(np.meshgrid(indices, indices)).T.reshape(-1,2).T
        return torch.tensor(
            self.data[indices[0], indices[1]]).reshape(dim, dim)

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
    """Relation data in neighbord tuple form.
    
    Args:
        relations: A tuple (n, r) of relation data, where n is an
        array of neighor indices for each data point and r is an
        array of relation values. Both arrays must be of shape
        (num_points, num_neighbors).

    Attributes:
        data: The raw relation data.
    """

    def __init__(self, relations: Tuple[np.ndarray,np.ndarray]) -> None:

        if not _is_array_tuple(relations):
            raise ValueError(
                'Expected tuple of arrays (neighbors, relations).'
            )

        # sort entries by ascending relation values
        neighbors, rels = relations
        indices = rels.argsort()
        neighbors = np.take_along_axis(neighbors, indices, axis=1)
        rels = np.take_along_axis(rels, indices, axis=1)

        self.data = (neighbors, rels)

    def sub(self, indices: IndexList) -> torch.Tensor:
        return self.to_sparse_array().sub(indices)

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


def _is_square_array(rels: Rels) -> bool:
    result = False
    if isinstance(rels, np.ndarray):
        if len(rels.shape) == 2 and rels.shape[0] == rels.shape[1]:
            result = True    
    return result

def _is_square_tensor(rels: Rels) -> bool:
    result = False
    if isinstance(rels, torch.Tensor):
        if len(rels.shape) == 2 and rels.shape[0] == rels.shape[1]:
            result = True
    return result

def _is_triangular_array(rels: Rels) -> bool:
    result = False
    if isinstance(rels, np.ndarray) and len(rels.shape) == 1:
        try:
            squareform(rels)
        except:
            pass
        else:
            result = True    
    return result

def _is_triangular_tensor(rels: Rels) -> bool:
    result = False
    if isinstance(rels, torch.Tensor):
        try:
            squareform(rels.detach().numpy())
        except:
            pass
        else:
            result = True    
    return result

def _is_flat_array(rels: Rels) -> bool:
    result = False
    if isinstance(rels, np.ndarray):
        if len(rels.shape) == 1:
            result = True
    return result

def _is_flat_tensor(rels: Rels) -> bool:
    result = False
    if isinstance(rels, torch.Tensor):
        if len(rels.shape) == 1:
            result = True
    return result

def _is_square_sparse(rels: Rels) -> bool:
    result = False
    if isinstance(rels, scipy.sparse.spmatrix):
        if (len(rels.shape) == 2 and
            rels.shape[0] == rels.shape[1]):
            result = True    
    return result

def _is_array_tuple(rels: Rels) -> bool:
    result = False
    if isinstance(rels, tuple) and len(rels) == 2:
        if len(rels[0].shape) == 2 and rels[0].shape == rels[1].shape:
            result = True    
    return result

def _tensor_squareform(t: torch.Tensor) -> torch.Tensor:
    d = _get_orig_dim(len(t))
    matrix = torch.zeros((d, d),
        dtype=t.dtype,
        device=t.device
    )
    a, b = torch.triu_indices(d, d, offset=1)
    matrix[[a, b]] = t
    return matrix + matrix.T

@functools.cache
def _rowcol_to_triu_index(i: int, j: int, dim: int) -> int:
    if i < j:
        index = round(i * (dim - 1.5) + j - i**2 * 0.5 - 1)
        return index
    elif i > j:
        return _rowcol_to_triu_index(j, i, dim)
    else:
        return -1

@functools.cache
def _get_orig_dim(len_triu: int) -> int:
    return int(np.ceil(np.sqrt(len_triu * 2)))