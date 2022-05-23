import numpy as np
from scipy.optimize import root_scalar
from scipy.sparse import csr_matrix
from numba import jit
from typing import overload, Literal, Tuple, Union
from .utils import report
from .types import Tensor, Distances


class DissimilarityTransform():

    def __init__(self):
        pass

    def __call__(self, X: Tensor) -> Tensor:
        pass

class Identity(DissimilarityTransform):
    def __call__(self, X: Tensor) -> Tensor:
        return X

class PerplexityBased(DissimilarityTransform):
    
    def __init__(self,
        perp: float = 30,
        beta_upper_bound = 1e6,
        verbose: bool = False
        ) -> None:

        self.perplexity = perp
        self.beta_upper_bound = beta_upper_bound
        self.verbose = verbose

    def transform(self,
        distmat: Distances
        ) -> Tensor:

        # TODO: check argument and transform accordingly

        neighbors, p_ij = convert_dissimilarity_format(
            distmat,
            'sparse'
        )
        num_pts = len(neighbors)
        k = neighbors.shape[1]

        if self.verbose:
            report('Calculating probabilities.')

        for i in range(num_pts):
            beta = find_beta(p_ij[i], self.perplexity)
            p_ij[i] = p_i(p_ij[i], beta)
        row_indices = np.repeat(np.arange(num_pts), k-1)
        p = csr_matrix((p_ij.ravel(), (row_indices, neighbors.ravel())))

        # TODO: make symmetrization optional

        return 0.5*(p + p.transpose())

# TODO: implement (but move to dissimilarity.py ?)
# def __convert_distances(
#     distmat)

# TODO: implement proper numpy type hints

@jit
def entropy(
    dists: np.ndarray,
    beta: float) -> float:

    x = -dists * beta
    y = np.exp(x)
    ysum: float = y.sum()

    if ysum < 1e-50:
        result = -1.
    else:
        factor = - 1/(np.log(2.) * ysum)
        result = factor * ((y * x) - (y * np.log(ysum))).sum()
    
    return result

def p_i(
    dists: Tensor,
    beta: float) -> np.ndarray:

    x = - dists * beta
    y = np.exp(x)
    ysum = y.sum()

    return y / ysum

def find_beta(
    dists: Tensor,
    perp: float,
    upper_bound: float = 1e6
    ) -> float:
    return root_scalar(
        lambda b: entropy(dists, b) - np.log2(perp),
        bracket=(0.,upper_bound)
    ).root


@overload
def convert_dissimilarity_format(
    D: Distances,
    out_format: Literal['square']
    ) -> np.ndarray: ...

@overload
def convert_dissimilarity_format(
    D: Distances,
    out_format: Literal['triangular']
    ) -> np.ndarray: ...

@overload
def convert_dissimilarity_format(
    D: Distances,
    out_format: Literal['sparse']
    ) -> Tuple[np.ndarray, np.ndarray]: ...

@overload
def convert_dissimilarity_format(
    D: Distances,
    out_format: str
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: ...