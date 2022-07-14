import numpy as np
import torch
import scipy.optimize
import scipy.sparse
from numba import jit
from typing import TypeVar, overload

from paradime import relationdata as pdreld
from .utils import report
from .types import Tensor, Rels


class RelationTransform():
    """Base class for relation transforms.
    
    Custom transforms should subclass this class.
    """

    def __init__(self):
        pass

    def __call__(self,reldata: pdreld.RelationData) -> pdreld.RelationData:

        return self.transform(reldata)

    def transform(self, reldata: pdreld.RelationData) -> pdreld.RelationData:
        """Applies the transform to input data.
        
        Args:
            reldata: The :class:`RelationData` instance to be transformed.

        Returns:
            A :class:`RelationData` instance containing the transformed
                relation values.
        """

        raise NotImplementedError()

class Identity(RelationTransform):
    """A placeholder identity transform."""

    def transform(self, reldata: pdreld.RelationData) -> pdreld.RelationData:

        return reldata

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

    def transform(self, reldata: pdreld.RelationData) -> pdreld.RelationData:

        X = reldata.to_array_tuple().data

        neighbors: np.ndarray = X[0][:, 1:]
        num_pts, k = neighbors.shape
        p_ij = np.empty((num_pts, k), dtype=float)
        self.beta = np.empty(num_pts, dtype=float)

        # TODO: think about what should happen if relations
        #       do not include r(x_i, x_i) = 0

        if self.verbose:
            report('Calculating probabilities.')

        for i in range(num_pts):
            beta = _find_beta(
                X[1][i, 1:],
                self.perplexity,
                **self.kwargs
            )
            self.beta[i] = beta
            p_ij[i] = _p_i(X[1][i, 1:], beta)
        
        return pdreld.NeighborRelationTuple((
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
def _entropy(dists: np.ndarray, beta: float) -> float:
    x = - dists**2 * beta
    y = np.exp(x)
    ysum = y.sum()

    if ysum < 1e-50:
        result = -1.
    else:
        factor = - 1/(np.log(2.) * ysum)
        result = factor * ((y * x) - (y * np.log(ysum))).sum()
    
    return result


def _p_i(dists: np.ndarray, beta: float) -> np.ndarray:
    x = - dists**2 * beta
    y = np.exp(x)
    ysum = y.sum()

    return y / ysum

def _find_beta(dists: np.ndarray, perp: float, **kwargs) -> float:
    return scipy.optimize.root_scalar(
        lambda b: _entropy(dists, b) - np.log2(perp),
        **kwargs
    ).root


class Symmetrize(RelationTransform):
    """Symmetrizes the relation values.
    
    Args:
        subtract_product: Specifies which symmetrization routine to use.
            If set to false (default), a matrix M is symmetrized by
            calculating 1/2 * (M + M^T); if set to true, M is symmetrized
            by calculating M + M^T - M * M^T, where '*' is the element-wise
            (Hadamard) product.
    """

    def __init__(self, subtract_prodcut: bool = False):

        self.subtract_product = subtract_prodcut

    def transform(self, reldata: pdreld.RelationData) -> pdreld.RelationData:

        if isinstance(reldata, (
            pdreld.TriangularRelationArray,
            pdreld.TriangularRelationTensor
        )):
            return reldata
        elif isinstance(reldata, (
            pdreld.FlatRelationArray,
            pdreld.FlatRelationTensor
        )):
            raise ValueError(
                "Flat list of relations cannot be symmetrized."
            )
        else:
            if self.subtract_product:
                symmetrizer = _sym_subtract_product
            else:
                symmetrizer = _sym_plus_only 

            if isinstance(reldata, pdreld.NeighborRelationTuple):
                return symmetrizer(reldata.to_sparse_array())
            else:
                return symmetrizer(reldata)


def _sym_plus_only(
    reldata: pdreld.RelationData
) -> pdreld.RelationData:
    if isinstance(reldata, pdreld.SquareRelationArray):
        reldata.data = (0.5 * (reldata.data + reldata.data.T))
    elif isinstance(reldata, pdreld.SparseRelationArray):
        reldata.data = 0.5 * (reldata.data + reldata.data.transpose())
    elif isinstance(reldata, pdreld.SquareRelationTensor):
        reldata.data = 0.5 * (reldata.data + torch.t(reldata.data))
    else:
        raise TypeError("Expected tensor-type :class:`RelationData`.")
    return reldata

def _sym_subtract_product(
    reldata: pdreld.RelationData
) -> pdreld.RelationData:
    if isinstance(reldata, pdreld.SquareRelationArray):
        reldata.data = (reldata.data + reldata.data.T
            - reldata.data * reldata.data.T)
    elif isinstance(reldata, pdreld.SparseRelationArray):
        reldata.data = (reldata.data + reldata.data.transpose()
            - reldata.data.multiply(reldata.data.transpose()))
    elif isinstance(reldata, pdreld.SquareRelationTensor):
        reldata.data = (reldata.data + torch.t(reldata.data)
            - reldata.data * torch.t(reldata.data))
    else:
        raise TypeError("Expected tensor-type :class:`RelationData`.")
    return reldata


class NormalizeRows(RelationTransform):
    """Normalizes the relation value for each data point separately."""

    #TODO: include code of norm_rows in transform method
    #TODO: implement norm_rows for neighborhood tuple (easy)
    def transform(self, reldata: pdreld.RelationData) -> pdreld.RelationData:

        if isinstance(reldata, (
            pdreld.TriangularRelationArray,
            pdreld.TriangularRelationTensor
        )):
            return reldata
        elif isinstance(reldata, (
            pdreld.FlatRelationArray,
            pdreld.FlatRelationTensor
        )):
            raise ValueError(
                "Flat list of relations cannot be normalized."
            )
        else:
            if isinstance(reldata, pdreld.NeighborRelationTuple):
                return _norm_rows(reldata.to_sparse_array())
            else:
                return _norm_rows(reldata)


def _norm_rows(reldata: pdreld.RelationData) -> pdreld.RelationData:
    if isinstance(reldata, pdreld.SquareRelationArray):
        reldata.data /= reldata.data.sum(axis=1, keepdims=True)
    elif isinstance(reldata, pdreld.SparseRelationArray):
        norm_factors = 1/np.repeat(
            np.array(reldata.data.sum(axis=1)),
            reldata.data.getnnz(axis=1)
        )
        reldata.data = scipy.sparse.csr_matrix((
            norm_factors,
            reldata.data.nonzero()
        )).multiply(reldata.data)
    elif isinstance(reldata, pdreld.SquareRelationTensor):
        reldata.data /= reldata.data.sum(dim=1, keepdim=True)
    else:
        raise TypeError('Expected tensor-type argument.')
    return reldata

#TODO: rewrite like other methods
class Normalize(RelationTransform):
    """Normalizes all relations."""

    def transform(self,
        reldat: pdreld.RelationData
        ) -> pdreld.RelationData:

        if not isinstance(X, pdreld.RelationData):
            X = pdreld.relation_factory(X)
        elif isinstance(X, pdreld.NeighborRelationTuple):
            X = X.to_sparse_array()

        X.data = X.data / X.data.sum()

        return X