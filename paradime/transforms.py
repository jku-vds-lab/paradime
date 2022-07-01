import numpy as np
import torch
import scipy.optimize
import scipy.sparse
from numba import jit
from typing import TypeVar, overload, Literal, Tuple, Union, Any

from paradime import relationdata as pdrel
from .utils import report
from .types import Tensor, Rels


class RelationTransform():
    """Base class for relation transforms.
    
    Custom transforms should subclass this class.
    """

    def __init__(self):
        pass

    def __call__(self,
        X: Union[Rels, pdrel.RelationData]
        ) -> pdrel.RelationData:

        return self.transform(X)

    def transform(self,
        X: Union[Rels, pdrel.RelationData]
        ) -> pdrel.RelationData:
        """Applies the transform to input data.
        
        Args:
            X: A :class:`RelationData` instance or raw input data as
                accepted by :func:`paradime.relationdata.relation_factory`.

        Returns:
            A :class:`RelationData` instance containing the transformed
                relation values.
        """

        raise NotImplementedError()

class Identity(RelationTransform):
    """A placeholder identity transform."""

    def transform(self,
        X: Union[Rels, pdrel.RelationData]
        ) -> pdrel.RelationData:

        if isinstance(X, pdrel.RelationData):
            return X
        else:
            return pdrel.relation_factory(X)

class PerplexityBased(RelationTransform):
    """Applies a perplexity-based transformation to the relation values.

    The relation values are rescaled using Guassian kernels. For each data
    point, the kernel width is determined by comparing the entropy of the
    relation values to the binary logarithm of the specified perplexity.
    This is the relation transform used by t-SNE.

    Args:
        perplexity: The desired perplexity, which can be understood as
            a smooth measure of nearest neighbors.
        verbose: Verbosity toggle.
        **kwargs: Passed on to :func:`scipy.optimize.root_scalar`, which
            determines the kernel widths. By default, this is set to use a
            bracket of [0.01, 1.] for the root search.    
    """
    
    def __init__(self,
        perplexity: float = 30,
        verbose: bool = False,
        **kwargs # passed on to root_scalar
        ) -> None:

        self.perplexity = perplexity
        self.kwargs = kwargs
        if not self.kwargs: # check if emtpy
            # self.kwargs['x0'] = 0.1
            # self.kwargs['x1'] = 1.0
            self.kwargs['bracket'] = [0.01, 1.]

        self.verbose = verbose

    def transform(self,
        X: Union[Rels, pdrel.RelationData]
        ) -> pdrel.RelationData:

        if isinstance(X, pdrel.RelationData):
            X = X.to_array_tuple()
        else:
            X = pdrel.relation_factory(X).to_array_tuple()

        neighbors = X.data[0][:, 1:]
        num_pts, k = neighbors.shape
        p_ij = np.empty((num_pts, k), dtype=float)
        self.beta = np.empty(num_pts, dtype=float)

        # TODO: think about what should happen if relations
        #       do not include r(x_i, x_i) = 0

        if self.verbose:
            report('Calculating probabilities.')

        for i in range(num_pts):
            beta = _find_beta(
                X.data[1][i, 1:],
                self.perplexity,
                **self.kwargs
            )
            self.beta[i] = beta
            p_ij[i] = _p_i(X.data[1][i, 1:], beta)
        
        return pdrel.NeighborRelationTuple((
            neighbors,
            p_ij
        ))
        # row_indices = np.repeat(np.arange(num_pts), k-1)
        # p = scipy.sparse.csr_matrix((
        #     p_ij.ravel(),
        #     (row_indices, neighbors.ravel())
        # ))

        # return prdmad.SparseAffinityArray(p)

@jit
def _entropy(
    dists: np.ndarray,
    beta: float) -> float:

    x = - dists**2 * beta
    y = np.exp(x)
    ysum = y.sum()

    if ysum < 1e-50:
        result = -1.
    else:
        factor = - 1/(np.log(2.) * ysum)
        result = factor * ((y * x) - (y * np.log(ysum))).sum()
    
    return result


def _p_i(
    dists: np.ndarray,
    beta: float) -> np.ndarray:

    x = - dists**2 * beta
    y = np.exp(x)
    ysum = y.sum()

    return y / ysum

def _find_beta(
    dists: np.ndarray,
    perp: float,
    **kwargs
    ) -> float:
    return scipy.optimize.root_scalar(
        lambda b: _entropy(dists, b) - np.log2(perp),
        **kwargs
    ).root


class Symmetrize(RelationTransform):
    """Symmetrizes the relation values.
    
    Args:
        impl: Specifies which symmetrization routine to use.
            Allowed values are `'tsne'` and `'umap'`.
    """

    def __init__(self, impl: Literal['tsne', 'umap']):

        self.impl = impl

    def transform(self,
        X: Union[Rels, pdrel.RelationData]
        ) -> pdrel.RelationData:

        if not isinstance(X, pdrel.RelationData):
            X = pdrel.relation_factory(X)
        elif isinstance(X, pdrel.NeighborRelationTuple):
            X = X.to_sparse_array()

        if self.impl == 'tsne':
            symmetrizer = _symm_tsne
        elif self.impl == 'umap':
            symmetrizer = _symm_umap
        else:
            raise ValueError('Expected specifier to be "umap" or "tsne".')

        X.data = symmetrizer(X.data)

        return X


@overload
def _symm_tsne(p: np.ndarray) -> np.ndarray:
    ...
@overload
def _symm_tsne(p: torch.Tensor) -> torch.Tensor:
    ...
@overload
def _symm_tsne(p: scipy.sparse.spmatrix) -> scipy.sparse.spmatrix:
    ...

def _symm_tsne(p: Tensor) -> Tensor:
    if isinstance(p, np.ndarray):
        return 0.5 * (p + p.T)
    elif isinstance(p, scipy.sparse.spmatrix):
        return 0.5 * (p + p.transpose())
    elif isinstance(p, torch.Tensor):
        return 0.5 * (p + torch.t(p))
    else:
        raise TypeError('Expected tensor-type argument.')


@overload
def _symm_umap(p: np.ndarray) -> np.ndarray:
    ...
@overload
def _symm_umap(p: torch.Tensor) -> torch.Tensor:
    ...
@overload
def _symm_umap(p: scipy.sparse.spmatrix) -> scipy.sparse.spmatrix:
    ...

def _symm_umap(p: Tensor) -> Tensor:
    if isinstance(p, np.ndarray):
        return p + p.T - (p * p.T)
    elif isinstance(p, scipy.sparse.spmatrix):
        return p + p.transpose() - (p.multiply(p.transpose()))
    elif isinstance(p, torch.Tensor):
        return p + torch.t(p) - (p * torch.t(p))
    else:
        raise TypeError('Expected tensor-type argument.')


class NormalizeRows(RelationTransform):
    """Normalizes the relation value for each data point separately."""

    def transform(self,
        X: Union[Rels, pdrel.RelationData]
        ) -> pdrel.RelationData:

        if not isinstance(X, pdrel.RelationData):
            X = pdrel.relation_factory(X)
        elif isinstance(X, pdrel.NeighborRelationTuple):
            X = X.to_sparse_array()

        X.data = _norm_rows(X.data)

        return X


@overload
def _norm_rows(p: np.ndarray) -> np.ndarray:
    ...
@overload
def _norm_rows(p: torch.Tensor) -> torch.Tensor:
    ...
@overload
def _norm_rows(p: scipy.sparse.spmatrix) -> scipy.sparse.spmatrix:
    ...

def _norm_rows(p: Tensor) -> Tensor:
    if isinstance(p, np.ndarray):
        return p / p.sum(axis=1, keepdims=True)
    elif isinstance(p, scipy.sparse.spmatrix):
        norm_factors = 1/np.repeat(
            np.array(p.sum(axis=1)),
            p.getnnz(axis=1)
        )
        return scipy.sparse.csr_matrix((
            norm_factors,
            p.nonzero()
        )).multiply(p)
    elif isinstance(p, torch.Tensor):
        return p / p.sum(dim=1, keepdim=True)
    else:
        raise TypeError('Expected tensor-type argument.')


class Normalize(RelationTransform):
    """Normalizes all relations."""

    def transform(self,
        X: Union[Rels, pdrel.RelationData]
        ) -> pdrel.RelationData:

        if not isinstance(X, pdrel.RelationData):
            X = pdrel.relation_factory(X)
        elif isinstance(X, pdrel.NeighborRelationTuple):
            X = X.to_sparse_array()

        X.data = X.data / X.data.sum()

        return X