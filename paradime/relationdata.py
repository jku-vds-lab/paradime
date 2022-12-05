"""Relation data containers for ParaDime.

The :mod:`paradime.relationdata` module implements container classes for
various formats of relation data. The relation data containers are used by the
different :class:`paradime.relations.Relations` (see :mod:`paradime.relations`)
and :class:`paradime.transforms.RelationTransform` (see
:mod:`paradime.transforms`).
"""

import itertools
from typing import Optional, Literal
import warnings

import numpy as np
import scipy.sparse
from scipy.spatial import distance
import torch

from paradime import utils
from paradime.types import IndexList, Rels


class RelationData(utils._ReprMixin):
    """Base class for storing relations between data points."""

    def __init__(self):
        self._data = None

    @property
    def data(self) -> Rels:
        return self._data

    @data.setter
    def data(self, relations: Rels) -> None:
        self._data = relations

    def sub(self, indices: IndexList) -> torch.Tensor:
        """Subsamples the relation matrix based on item indices.

        Intended to be used for batch-wise subsampling of global relations.

        Args:
            indices: A flat list of item indices.

        Returns:
            A square PyTorch tensor consisting of all relations
            between items with the given indices.
        """
        raise NotImplementedError(
            f"Matrix subsampling not implemented for {type(self).__name__}."
        )

    def to_flat_array(self) -> "FlatRelationArray":
        """Converts the relations to a
        :class:`paradime.relationdata.FlatRelationArray`.

        Returns:
            The converted relations.
        """
        raise NotImplementedError(
            "Conversion to flat array not "
            f"implemented for {type(self).__name__}."
        )

    def to_flat_tensor(self) -> "FlatRelationTensor":
        """Converts the relations to a
        :class:`paradime.relationdata.FlatRelationTensor`.

        Returns:
            The converted relations.
        """
        raise NotImplementedError(
            "Conversion to flat tensor not "
            f"implemented for {type(self).__name__}."
        )

    def to_square_array(self) -> "SquareRelationArray":
        """Converts the relations to a
        :class:`paradime.relationdata.SquareRelationArray`.

        Returns:
            The converted relations.
        """
        raise NotImplementedError(
            "Conversion to square array not "
            f"implemented for {type(self).__name__}."
        )

    def to_square_tensor(self) -> "SquareRelationTensor":
        """Converts the relations to a
        :class:`paradime.relationdata.SquareRelationTensor`.

        Returns:
            The converted relations.
        """
        raise NotImplementedError(
            "Conversion to square tensor not "
            f"implemented for {type(self).__name__}."
        )

    def to_triangular_array(self) -> "TriangularRelationArray":
        """Converts the relations to a
        :class:`paradime.relationdata.TriangularRelationArray`.

        Returns:
            The converted relations.
        """
        raise NotImplementedError(
            "Conversion to triangular array not "
            f"implemented for {type(self).__name__}."
        )

    def to_triangular_tensor(self) -> "TriangularRelationTensor":
        """Converts the relations to a
        :class:`paradime.relationdata.TriangularRelationTensor`.

        Returns:
            The converted relations.
        """
        raise NotImplementedError(
            "Conversion to triangular tensor not "
            f"implemented for {type(self).__name__}."
        )

    def to_neighbor_tuple(self) -> "NeighborRelationTuple":
        """Converts the relations to a
        :class:`paradime.relationdata.NeighborRelationTuple`.

        Returns:
            The converted relations.
        """
        raise NotImplementedError(
            "Conversion to neighbor relation tuple not "
            f"implemented for {type(self).__name__}."
        )

    def to_sparse_array(self) -> "SparseRelationArray":
        """Converts the relations to a
        :class:`paradime.relationdata.SparseRelationArray`.

        Returns:
            The converted relations.
        """
        raise NotImplementedError(
            "Conversion to sparse array not "
            f"implemented for {type(self).__name__}."
        )


def relation_factory(
    relations: Rels,
    force_flat: bool = False,
) -> RelationData:
    """Creates a :class:`paradime.relationdata.RelationData` object from a
    variety of input formats.

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
        A :class:`paradime.relationdata.RelationData` object with a subclass
        depending on the input format.
    """

    if _is_square_array(relations):
        rd = SquareRelationArray(relations)  # type: ignore
    elif _is_square_tensor(relations):
        rd = SquareRelationTensor(relations)  # type: ignore
    elif _is_square_sparse(relations):
        rd = SparseRelationArray(relations)  # type: ignore
    elif _is_triangular_array(relations) and not force_flat:
        rd = TriangularRelationArray(relations)  # type: ignore
    elif _is_triangular_tensor(relations) and not force_flat:
        rd = TriangularRelationTensor(relations)  # type: ignore
    elif _is_flat_array(relations):
        rd = FlatRelationArray(relations)  # type: ignore
    elif _is_flat_tensor(relations):
        rd = FlatRelationTensor(relations)  # type: ignore
    elif _is_array_tuple(relations):
        rd = NeighborRelationTuple(relations)  # type: ignore
    else:
        raise TypeError(f"Input type not supported by {RelationData.__name__}.")

    return rd


class FlatRelationArray(RelationData):
    """Relation data in the form of a flat array of individual relations.

    Args:
        relations: A flat Numpy array of relation values.

    Attributes:
        data: The raw relation data.
    """

    def __init__(self, relations: np.ndarray):

        if not _is_flat_array(relations):
            raise ValueError("Expected vector-form array.")

        self._data = relations

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, relations: np.ndarray) -> None:
        self._data = relations

    def to_flat_array(self) -> "FlatRelationArray":
        return self

    def to_flat_tensor(self) -> "FlatRelationTensor":
        return FlatRelationTensor(torch.tensor(self.data))


class FlatRelationTensor(RelationData):
    """Relation data in the form of a flat tensor of individual relations.

    Args:
        relations: A flat PyTorch tensor of relation values.

    Attributes:
        data: The raw relation data.
    """

    def __init__(self, relations: torch.Tensor):

        if not _is_flat_tensor(relations):
            raise ValueError("Expected vector-form tensor.")

        self._data = relations

    @property
    def data(self) -> torch.Tensor:
        return self._data

    @data.setter
    def data(self, relations: torch.Tensor) -> None:
        self._data = relations

    def to_flat_array(self) -> "FlatRelationArray":
        return FlatRelationArray(self.data.detach().numpy())

    def to_flat_tensor(self) -> "FlatRelationTensor":
        return self


class SquareRelationArray(RelationData):
    """Relation data in the form of a square array.

    Args:
        relations: A square Numpy array of relation values.

    Attributes:
        data: The raw relation data.
    """

    def __init__(self, relations: np.ndarray):

        if not _is_square_array(relations):
            raise ValueError("Expected square array.")

        self._data = relations

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, relations: np.ndarray) -> None:
        self._data = relations

    def sub(self, indices: IndexList) -> torch.Tensor:
        dim = len(indices)
        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu().numpy()
        indices = np.array(np.meshgrid(indices, indices)).T.reshape(-1, 2).T
        return torch.tensor(
            np.reshape(self.data[indices[0], indices[1]], (dim, dim))
        )

    def to_square_array(self) -> "SquareRelationArray":
        return self

    def to_square_tensor(self) -> "SquareRelationTensor":
        return SquareRelationTensor(torch.tensor(self.data))

    def to_triangular_array(self) -> "TriangularRelationArray":
        return TriangularRelationArray(distance.squareform(self.data))

    def to_triangular_tensor(self) -> "TriangularRelationTensor":
        return TriangularRelationTensor(
            torch.tensor(distance.squareform(self.data))
        )

    def to_sparse_array(self) -> "SparseRelationArray":
        return SparseRelationArray(scipy.sparse.csr_matrix(self.data))

    def to_neighbor_tuple(self) -> "NeighborRelationTuple":
        return NeighborRelationTuple(
            (
                np.tile(np.arange(self.data.shape[0]), (self.data.shape[0], 1)),
                self.data,
            )
        )


class SquareRelationTensor(RelationData):
    """Relation data in the form of a square tensor.

    Args:
        relations: A square PyTorch tensor of relation values.

    Attributes:
        data: The raw relation data.
    """

    def __init__(self, relations: torch.Tensor):

        if not _is_square_tensor(relations):
            raise ValueError("Expected square tensor.")

        self._data = relations

    @property
    def data(self) -> torch.Tensor:
        return self._data

    @data.setter
    def data(self, relations: torch.Tensor) -> None:
        self._data = relations

    def sub(self, indices: IndexList) -> torch.Tensor:
        dim = len(indices)
        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu().numpy()
        indices = np.array(np.meshgrid(indices, indices)).T.reshape(-1, 2).T
        return self.data[indices[0], indices[1]].reshape(dim, dim)

    def to_square_array(self) -> "SquareRelationArray":
        return SquareRelationArray(self.data.detach().cpu().numpy())

    def to_square_tensor(self) -> "SquareRelationTensor":
        return self

    def to_triangular_array(self) -> "TriangularRelationArray":
        return TriangularRelationArray(
            distance.squareform(self.data.detach().numpy())
        )

    def to_triangular_tensor(self) -> "TriangularRelationTensor":
        i, j = torch.triu_indices(
            self.data.shape[0], self.data.shape[1], offset=1
        )
        return TriangularRelationTensor(self.data[i, j])

    def to_sparse_array(self) -> "SparseRelationArray":
        return SparseRelationArray(
            scipy.sparse.csr_matrix(self.data.detach().numpy())
        )

    def to_neighbor_tuple(self) -> "NeighborRelationTuple":
        rels = self.data.detach().numpy()
        return NeighborRelationTuple(
            (np.tile(np.arange(rels.shape[0]), (rels.shape[0], 1)), rels)
        )


class TriangularRelationArray(RelationData):
    """Relation data in 'triangular' vector-form.

    Args:
        relations: A Numpy array of relation values, as accepted by
            :func:`scipy.spatial.distance.squareform`.

    Attributes:
        data: The raw relation data.
    """

    def __init__(self, relations: np.ndarray):

        if not _is_triangular_array(relations):
            raise ValueError("Expected vector-form relation array.")

        self._data = relations

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, relations: np.ndarray) -> None:
        self._data = relations

    def sub(self, indices: IndexList) -> torch.Tensor:
        dim = utils.convert.triu_to_square_dim(len(self.data))
        combos = itertools.combinations(indices, 2)
        return torch.tensor(
            distance.squareform(
                np.array(
                    [
                        self.data[utils.convert.rowcol_to_triu_index(i, j, dim)]
                        for i, j in combos
                    ]
                )
            )
        )

    def to_square_array(self) -> "SquareRelationArray":
        return SquareRelationArray(distance.squareform(self.data))

    def to_square_tensor(self) -> "SquareRelationTensor":
        return SquareRelationTensor(
            torch.tensor(distance.squareform(self.data))
        )

    def to_triangular_array(self) -> "TriangularRelationArray":
        return self

    def to_triangular_tensor(self) -> "TriangularRelationTensor":
        return TriangularRelationTensor(torch.tensor(self.data))

    def to_sparse_array(self) -> "SparseRelationArray":
        return SparseRelationArray(
            scipy.sparse.csr_matrix(distance.squareform(self.data))
        )

    def to_neighbor_tuple(self) -> "NeighborRelationTuple":
        return self.to_square_array().to_neighbor_tuple()


class TriangularRelationTensor(RelationData):
    """Relation data in 'triangular' vector-form.

    Args:
        relations: A PyTorch tensor of relation values, with a shape as
            accepted by :func:`scipy.spatial.distance.squareform`.

    Attributes:
        data: The raw relation data.
    """

    def __init__(self, relations: torch.Tensor):

        if not _is_triangular_tensor(relations):
            raise ValueError("Expected vector-form relation tensor.")

        self._data = relations

    @property
    def data(self) -> torch.Tensor:
        return self._data

    @data.setter
    def data(self, relations: torch.Tensor) -> None:
        self._data = relations

    def sub(self, indices: IndexList) -> torch.Tensor:
        dim = utils.convert.triu_to_square_dim(len(self.data))
        combos = itertools.combinations(indices, 2)
        return _tensor_squareform(
            torch.tensor(
                [
                    self.data[utils.convert.rowcol_to_triu_index(i, j, dim)]
                    for i, j in combos
                ]
            )
        )

    def to_square_array(self) -> "SquareRelationArray":
        return SquareRelationArray(
            distance.squareform(self.data.detach().numpy())
        )

    def to_square_tensor(self) -> "SquareRelationTensor":
        # get dimensions of square matrix
        d = utils.convert.triu_to_square_dim(len(self.data))
        matrix = torch.zeros(
            (d, d), dtype=self.data.dtype, device=self.data.device
        )
        a, b = torch.triu_indices(d, d, offset=1)
        matrix[[a, b]] = self.data
        matrix = matrix + matrix.T
        return SquareRelationTensor(matrix)

    def to_triangular_array(self) -> "TriangularRelationArray":
        return TriangularRelationArray(self.data.detach().numpy())

    def to_triangular_tensor(self) -> "TriangularRelationTensor":
        return self

    def to_sparse_array(self) -> "SparseRelationArray":
        return self.to_triangular_array().to_sparse_array()

    def to_neighbor_tuple(self) -> "NeighborRelationTuple":
        return self.to_square_array().to_neighbor_tuple()


class SparseRelationArray(RelationData):
    """Relation data in sparse array form.

    Args:
        relations: A square, sparse Scipy array of relation values.

    Attributes:
        data: The raw relation data.
    """

    def __init__(self, relations: scipy.sparse.spmatrix):

        if not _is_square_sparse(relations):
            raise ValueError("Expected vector-form relation tensor.")

        self._data = relations

    @property
    def data(self) -> scipy.sparse.spmatrix:
        return self._data

    @data.setter
    def data(self, relations: scipy.sparse.spmatrix) -> None:
        self._data = relations

    def sub(self, indices: IndexList) -> torch.Tensor:
        dim = len(indices)
        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu().numpy()
        indices = np.array(np.meshgrid(indices, indices)).T.reshape(-1, 2).T
        return torch.tensor(self.data[indices[0], indices[1]]).reshape(dim, dim)

    def to_square_array(self) -> "SquareRelationArray":
        return SquareRelationArray(self.data.toarray())

    def to_square_tensor(self) -> "SquareRelationTensor":
        return SquareRelationTensor(torch.tensor(self.data.toarray()))

    def to_triangular_array(self) -> "TriangularRelationArray":
        indices = np.triu_indices(self.data.shape[0], k=1)
        return TriangularRelationArray(self.data[indices].A[0])

    def to_triangular_tensor(self) -> "TriangularRelationTensor":
        indices = np.triu_indices(self.data.shape[0], k=1)
        return TriangularRelationTensor(torch.tensor(self.data[indices].A[0]))

    def to_sparse_array(self) -> "SparseRelationArray":
        return self

    def to_neighbor_tuple(self) -> "NeighborRelationTuple":
        return self.to_square_array().to_neighbor_tuple()


class NeighborRelationTuple(RelationData):
    """Relation data in neighborhood tuple form.

    Args:
        relations: A tuple (n, r) of relation data, where n is an array of
            neighor indices for each data point and r is an array of relation
            values. Both arrays must be of shape (num_points, num_neighbors).
        sort: Sorting option. If None is passed (default), values are kept
            as is. Otherwise, values for each item are sorted either in
            ``'ascending'`` or ``'descending'`` order.

    Attributes:
        data: The raw relation data.
    """

    def __init__(
        self,
        relations: tuple[np.ndarray, np.ndarray],
        sort: Optional[Literal["ascending", "descending"]] = None,
    ):

        if not _is_array_tuple(relations):
            raise ValueError("Expected tuple of arrays (neighbors, relations).")

        neighbors, rels = relations

        if sort is None:
            pass
        elif sort == "ascending":
            indices = rels.argsort()
            neighbors = np.take_along_axis(neighbors, indices, axis=1)
            rels = np.take_along_axis(rels, indices, axis=1)
        elif sort == "descending":
            indices = rels.argsort()[:, ::-1]
            neighbors = np.take_along_axis(neighbors, indices, axis=1)
            rels = np.take_along_axis(rels, indices, axis=1)
        else:
            raise ValueError(
                f"Unknown sorting option {sort}. Only None, 'ascending' "
                "or 'descending are supported."
            )

        self._data: tuple[np.ndarray, np.ndarray] = (neighbors, rels)

    @property
    def data(self) -> tuple[np.ndarray, np.ndarray]:
        return self._data

    @data.setter
    def data(self, relations: tuple[np.ndarray, np.ndarray]) -> None:
        self._data = relations

    def _where_self_relations(self) -> np.ndarray:
        relations = self.data[0]
        return relations == np.arange(len(relations))[:, None]

    def _has_all_self_relations(self) -> bool:
        return bool(np.all(np.any(self._where_self_relations(), axis=1)))

    def _remove_self_relations(self) -> None:
        if not self._has_all_self_relations():
            warnings.warn(
                "Neighbor relation tuple does not include all self-relations. "
                "Removal would lead to ragged array and is not performed."
            )
        else:
            mask = np.logical_not(self._where_self_relations())
            self.data = (
                np.stack([i[j] for i, j in zip(self.data[0], mask)]),
                np.stack([i[j] for i, j in zip(self.data[1], mask)]),
            )

    def sub(self, indices: IndexList) -> torch.Tensor:
        return self.to_sparse_array().sub(indices)

    def to_square_array(self) -> "SquareRelationArray":
        return self.to_sparse_array().to_square_array()

    def to_square_tensor(self) -> "SquareRelationTensor":
        return self.to_sparse_array().to_square_tensor()

    def to_triangular_array(self) -> "TriangularRelationArray":
        return self.to_sparse_array().to_triangular_array()

    def to_triangular_tensor(self) -> "TriangularRelationTensor":
        return self.to_sparse_array().to_triangular_tensor()

    def to_sparse_array(self) -> "SparseRelationArray":
        neighbors, rels = self.data
        row_indices = np.repeat(
            np.arange(neighbors.shape[0]), neighbors.shape[1]
        )
        matrix = scipy.sparse.csr_matrix(
            (rels.ravel(), (row_indices, neighbors.ravel()))
        )
        return SparseRelationArray(matrix)

    def to_neighbor_tuple(self) -> "NeighborRelationTuple":
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
    if isinstance(rels, np.ndarray):
        s = rels.shape
        if len(s) == 1:
            d = int(np.ceil(np.sqrt(s[0] * 2)))
            if d * (d - 1) == s[0] * 2:
                result = True
    return result


def _is_triangular_tensor(rels: Rels) -> bool:
    result = False
    if isinstance(rels, torch.Tensor):
        s = rels.shape
        if len(s) == 1:
            d = int(np.ceil(np.sqrt(s[0] * 2)))
            if d * (d - 1) == s[0] * 2:
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
        if len(rels.shape) == 2 and rels.shape[0] == rels.shape[1]:
            result = True
    return result


def _is_array_tuple(rels: Rels) -> bool:
    result = False
    if isinstance(rels, tuple) and len(rels) == 2:
        if len(rels[0].shape) == 2 and rels[0].shape == rels[1].shape:
            if rels[0].max() < len(rels[0]):
                result = True
    return result


def _tensor_squareform(t: torch.Tensor) -> torch.Tensor:
    d = utils.convert.triu_to_square_dim(len(t))
    matrix = torch.zeros((d, d), dtype=t.dtype, device=t.device)
    a, b = torch.triu_indices(d, d, offset=1)
    matrix[[a, b]] = t
    return matrix + matrix.T
