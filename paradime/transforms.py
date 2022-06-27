import numpy as np
import torch
import scipy.optimize
import scipy.sparse
from numba import jit
from typing import TypeVar, overload, Literal, Tuple, Union, Any

from paradime import affinitydata as prdmad
from .utils import report
from .types import Tensor, Diss, Symm


class AffinityTransform():

    def __init__(self):
        pass

    def __call__(self,
        X: Union[Diss, prdmad.AffinityData]
        ) -> prdmad.AffinityData:
        pass

class Identity(AffinityTransform):
    def __call__(self,
        X: Union[Diss, prdmad.AffinityData]
        ) -> prdmad.AffinityData:

        if isinstance(X, prdmad.AffinityData):
            return X
        else:
            return prdmad.affinity_factory(X)

class PerplexityBased(AffinityTransform):
    
    def __init__(self,
        perp: float = 30,
        verbose: bool = False,
        **kwargs # passed on to root_scalar
        ) -> None:

        self.perplexity = perp
        self.kwargs = kwargs
        if not self.kwargs: # check if emtpy
            self.kwargs['x0'] = 0.
            self.kwargs['x1'] = 1.

        self.verbose = verbose

    def __call__(self,
        X: Union[Diss, prdmad.AffinityData]
        ) -> prdmad.AffinityData:

        return self.transform(X)

    def transform(self,
        X: Union[Diss, prdmad.AffinityData]
        ) -> prdmad.AffinityData:

        if isinstance(X, prdmad.AffinityData):
            X = X.to_array_tuple()
        else:
            X = prdmad.affinity_factory(X).to_array_tuple()

        neighbors = X.diss[0][:, 1:]
        num_pts, k = neighbors.shape
        p_ij = np.empty((num_pts, k), dtype=float)
        self.beta = np.empty(num_pts, dtype=float)

        # TODO: think about what should happen if affinities
        #       do not include d(x_i, x_i) = 0

        if self.verbose:
            report('Calculating probabilities.')

        for i in range(num_pts):
            beta = find_beta(
                X.diss[1][i, 1:],
                self.perplexity,
                **self.kwargs
            )
            self.beta[i] = beta
            p_ij[i] = p_i(X.diss[1][i, 1:], beta)
        
        return prdmad.AffinityTuple((
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
def entropy(
    dists: np.ndarray,
    beta: float) -> float:

    x = dists**2 * beta
    y = np.exp(x)
    ysum = y.sum()

    if ysum < 1e-50:
        result = -1.
    else:
        factor = - 1/(np.log(2.) * ysum)
        result = factor * ((y * x) - (y * np.log(ysum))).sum()
    
    return result


def p_i(
    dists: np.ndarray,
    beta: float) -> np.ndarray:

    x = - dists**2 * beta
    y = np.exp(x)
    ysum = y.sum()

    return y / ysum

def find_beta(
    dists: np.ndarray,
    perp: float,
    **kwargs
    ) -> float:
    return scipy.optimize.root_scalar(
        lambda b: entropy(dists, b) - np.log2(perp),
        **kwargs
    ).root


class Symmetrize(AffinityTransform):

    def __init__(self, impl: Literal['tsne', 'umap']):

        self.impl = impl

    def transform(self,
        X: Union[Diss, prdmad.AffinityData]
        ) -> prdmad.AffinityData:

        if not isinstance(X, prdmad.AffinityData):
            X = prdmad.affinity_factory(X)
        elif isinstance(X, prdmad.AffinityTuple):
            X = X.to_sparse_array()

        if self.impl == 'tsne':
            symmetrizer = symm_tsne
        elif self.impl == 'umap':
            symmetrizer = symm_umap
        else:
            raise ValueError('Expected specifier to be "umap" or "tsne".')

        X.diss = symmetrizer(X.diss)

        return X




@overload
def symm_tsne(p: np.ndarray) -> np.ndarray:
    ...
@overload
def symm_tsne(p: torch.Tensor) -> torch.Tensor:
    ...
@overload
def symm_tsne(p: scipy.sparse.spmatrix) -> scipy.sparse.spmatrix:
    ...

def symm_tsne(p: Tensor) -> Tensor:
    if isinstance(p, np.ndarray):
        return 0.5 * (p + p.T)
    elif isinstance(p, scipy.sparse.spmatrix):
        return 0.5 * (p + p.transpose())
    elif isinstance(p, torch.Tensor):
        return 0.5 * (p + torch.t(p))
    else:
        raise TypeError('Expected tensor-type argument.')


@overload
def symm_umap(p: np.ndarray) -> np.ndarray:
    ...
@overload
def symm_umap(p: torch.Tensor) -> torch.Tensor:
    ...
@overload
def symm_umap(p: scipy.sparse.spmatrix) -> scipy.sparse.spmatrix:
    ...

def symm_umap(p: Tensor) -> Tensor:
    if isinstance(p, np.ndarray):
        return p + p.T - (p * p.T)
    elif isinstance(p, scipy.sparse.spmatrix):
        return p + p.transpose() - (p.multiply(p.transpose()))
    elif isinstance(p, torch.Tensor):
        return p + torch.t(p) - (p * torch.t(p))
    else:
        raise TypeError('Expected tensor-type argument.')
