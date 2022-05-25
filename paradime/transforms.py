import numpy as np
import torch
from psutil import disk_io_counters
from scipy.optimize import root_scalar
from scipy.sparse import csr_matrix, spmatrix
from numba import jit
from typing import TypeVar, overload, Literal, Tuple, Union, Any
from nptyping import NDArray, Shape, Float
from .dissimilarity import DissimilarityData, DissimilarityTuple
from .utils import report
from .types import Tensor, Diss, Symm


class DissimilarityTransform():

    def __init__(self):
        pass

    def __call__(self,
        X: Union[Diss, DissimilarityData]
        ) -> DissimilarityData:
        pass

class Identity(DissimilarityTransform):
    def __call__(self,
        X: Union[Diss, DissimilarityData]
        ) -> DissimilarityData:

        if isinstance(X, DissimilarityData):
            return X
        else:
            return DissimilarityData(X)

class PerplexityBased(DissimilarityTransform):
    
    def __init__(self,
        perp: float = 30,
        beta_upper_bound: float = 1e6,
        verbose: bool = False,
        symmetrize: Symm = 'tsne'
        ) -> None:

        self.perplexity = perp
        self.beta_upper_bound = beta_upper_bound
        self.symmetrize = symmetrize
        self.verbose = verbose

    def __call__(self,
        X: Union[Diss, DissimilarityData]
        ) -> DissimilarityData:

        return self.transform(X)

    def transform(self,
        X: Union[Diss, DissimilarityData]
        ) -> Tensor:

        if isinstance(X, DissimilarityData):
            X = X.to_array_tuple()
        else:
            X = DissimilarityData(X).to_array_tuple()

        neighbors, p_ij = X.diss
        num_pts = len(neighbors)
        k = neighbors.shape[1]

        if self.verbose:
            report('Calculating probabilities.')

        for i in range(num_pts):
            beta = find_beta(p_ij[i], self.perplexity)
            p_ij[i] = p_i(p_ij[i], beta)
        row_indices = np.repeat(np.arange(num_pts), k-1)
        p = csr_matrix((p_ij.ravel(), (row_indices, neighbors.ravel())))

        if self.symmetrize is not None:
            if self.symmetrize == 'tsne':
                p = symm_tsne(p)
            elif self.symmetrize == 'umap':
                p = symm_umap(p)
            elif callable(self.symmetrize):
                p = self.symmetrize(p)

        return p

@jit
def entropy(
    dists: NDArray[Shape['*'], Float],
    beta: float) -> float:

    x = dists * beta
    y = np.exp(x)
    ysum = y.sum()

    if ysum < 1e-50:
        result = -1.
    else:
        factor = - 1/(np.log(2.) * ysum)
        result = factor * ((y * x) - (y * np.log(ysum))).sum()
    
    return result


def p_i(
    dists: NDArray[Shape['*'], Float],
    beta: float) -> NDArray[Shape['*'], Float]:

    x = - dists * beta
    y = np.exp(x)
    ysum = y.sum()

    return y / ysum

def find_beta(
    dists: NDArray[Shape['*'], Float],
    perp: float,
    upper_bound: float = 1e6
    ) -> float:
    return root_scalar(
        lambda b: entropy(dists, b) - np.log2(perp),
        bracket=(0.,upper_bound)
    ).root


@overload
def symm_tsne(p: NDArray[Shape['Dim, Dim'], Any]
    ) -> NDArray[Shape['Dim, Dim'], Any]:
    ...
@overload
def symm_tsne(p: torch.Tensor) -> torch.Tensor:
    ...
@overload
def symm_tsne(p: spmatrix) -> spmatrix:
    ...

def symm_tsne(p: Tensor) -> Tensor:
    if isinstance(p, np.ndarray):
        return 0.5 * (p + p.T)
    elif isinstance(p, spmatrix):
        return 0.5 * (p + p.transpose())
    elif isinstance(p, torch.Tensor):
        return 0.5 * (p + torch.t(p))
    else:
        raise TypeError('Expected tensor-type argument.')

@overload
def symm_umap(p: NDArray[Shape['Dim, Dim'], Any]
    ) -> NDArray[Shape['Dim, Dim'], Any]:
    ...
@overload
def symm_umap(p: torch.Tensor) -> torch.Tensor:
    ...
@overload
def symm_umap(p: spmatrix) -> spmatrix:
    ...

def symm_umap(p: Tensor) -> Tensor:
    if isinstance(p, np.ndarray):
        return p + p.T - (p * p.T)
    elif isinstance(p, spmatrix):
        return p + p.transpose() - (p.multiply(p.transpose()))
    elif isinstance(p, torch.Tensor):
        return p + torch.t(p) - (p * torch.t(p))
    else:
        raise TypeError('Expected tensor-type argument.')
