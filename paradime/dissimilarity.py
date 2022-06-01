from datetime import datetime
import warnings
from typing import Union, Callable, Literal, Tuple, Any
import torch
import torch.nn.functional as F
import numpy as np
from pynndescent import NNDescent
import scipy.sparse
from scipy.spatial.distance import pdist, squareform

from paradime import dissimilaritydata as prdmdd
from paradime import transforms as prdmtf
from .types import Metric, Tensor, Diss
from .utils import report


class Dissimilarity():
    
    def __init__(self,
        transform: Union[Callable, prdmtf.DissimilarityTransform] = None
        ) -> None:
        
        self.transform = transform

    def compute_dissimilarities(self,
        X: Tensor = None,
        **kwargs) -> prdmdd.DissimilarityData:

        raise NotImplementedError

    def _transform(self,
        X: prdmdd.DissimilarityData
        ) -> prdmdd.DissimilarityData:

        if self.transform is None:
            return X
        else:
            return self.transform(X)


class Precomputed(Dissimilarity):

    def __init__(self,
        X: Tensor,
        transform: Union[Callable, prdmtf.DissimilarityTransform] = None
        ) -> None:

        super().__init__(
            transform = transform
        )

        self.dissimilarities = self._transform(
            prdmdd.dissimilarity_factory(X))

    def compute_dissimilarities(self,
        X: Tensor = None,
        **kwargs
        ) -> prdmdd.DissimilarityData:

        if X is not None:
            warnings.warn('Ignoring input for precomputed dissimilarity')
        
        return self.dissimilarities


class Exact(Dissimilarity):
 
    def __init__(self,
        metric: Metric = None,
        keep_result = True,
        transform: Union[Callable, prdmtf.DissimilarityTransform] = None,
        verbose: Union[int, bool] = False
        ) -> None:

        if metric is None:
            metric = 'euclidean'

        super().__init__(
            transform=transform
        )

        self.metric = metric
        self.keep_result = keep_result
        self.verbose = verbose

    def compute_dissimilarities(self,
        X: Tensor = None,
        **kwargs
        ) -> prdmdd.DissimilarityData:

        if X is None:
            raise ValueError(
                'Missing input for non-precomputed dissimilarity.'
            )

        X = _convert_input_to_numpy(X)

        if not hasattr(self, 'dissimilarities') or not self.keep_result:
            if self.verbose:
                report('Calculating pairwise distances.')
            self.dissimilarities = self._transform(
                prdmdd.dissimilarity_factory(pdist(X, metric=self.metric))
            )
        elif self.verbose:
            report('Using previously calculated distances.')

        return self.dissimilarities


class NeighborBased(Dissimilarity):

    def __init__(self,
        n_neighbors: int = None,
        metric: Metric = None,
        transform: Union[Callable, prdmtf.DissimilarityTransform] = None,
        verbose: Union[bool, int] = False
        ) -> None:

        super().__init__(
            transform=transform
        )

        self.n_neighbors = n_neighbors
        self.verbose = verbose
        self.metric = metric
    
    def compute_dissimilarities(self,
        X: Tensor = None,
        **kwargs
        ) -> prdmdd.DissimilarityData:

        if X is None:
            raise ValueError(
                'Missing input for non-precomputed dissimilarity.'
            )

        X = _convert_input_to_numpy(X)

        num_pts = X.shape[0]

        if self.n_neighbors is None:
            if isinstance(self.transform, prdmtf.PerplexityBased):
                self.n_neighbors = min(
                    num_pts - 1,
                    int(3 * self.transform.perplexity)
                )
            else:
                self.n_neighbors = int(0.1 * num_pts)
        else:
            if isinstance(self.transform, prdmtf.PerplexityBased):
                if self.n_neighbors < 3 * self.transform.perplexity:
                    warnings.warn(
                        f'Number of neighbors {self.n_neighbors} ' +
                        'smaller than three times perplexity' +
                        f'{self.transform.perplexity} of transform.'
                    )
        
        if self.verbose:
            report('Indexing nearest neighbors.')

        if self.metric is None:
            self.metric = 'euclidean'
        
        index = NNDescent(X,
            n_neighbors=self.n_neighbors,
            metric = self.metric
        )
        neighbors, distances = index.neighbor_graph

        self.dissimilarities = self._transform(
            prdmdd.dissimilarity_factory(
                (neighbors, distances)
            )
        )

        return self.dissimilarities


class Differentiable(Dissimilarity):

    def __init__(self,
        p: float = 2,
        metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
        transform: Union[Callable, prdmtf.DissimilarityTransform] = None,
        verbose: Union[int, bool] = False
        ) -> None:

        super().__init__(
            transform=transform
        )

        self.metric = metric
        self.metric_p = p

        self.verbose = verbose

    def compute_dissimilarities(self,
        X: Tensor = None,
        **kwargs
        ) -> prdmdd.DissimilarityData:

        if X is None:
            raise ValueError(
                'Missing input for non-precomputed dissimilarity.'
            )

        if not isinstance(X, torch.Tensor) or not X.requires_grad:
            warnings.warn(
                'Differentiable dissimilarity operating on tensor ' +
                'for which no gradients are computed.'
            )

        X = _convert_input_to_torch(X)

        # use memory-inefficient pdist to allow for arbitrary metrics
        # will break for large batches
        if self.metric is not None:
            n = X.shape[0]
            expanded = X.unsqueeze(1)
            # repeat entries n times
            tiled = torch.repeat_interleave(expanded, n, dim=1)
            # apply metric to pairs of items
            diss = self.metric(tiled, tiled.transpose(0, 1))
            self.dissimilarities = self._transform(
                prdmdd.SquareDissimilarityTensor(diss)
            )
        # otherwise use built-in torch method
        else:
            self.dissimilarities = self._transform(
                prdmdd.TriangularDissimilarityTensor(
                    F.pdist(X, p=self.metric_p)
                )
            )

        return self.dissimilarities



def _convert_input_to_numpy(
    X: Tensor) -> np.ndarray:
    
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    elif isinstance(X, scipy.sparse.spmatrix):
        X = X.toarray()
    elif isinstance(X, np.ndarray):
        pass
    else:
        raise TypeError(f'Input type {type(X)} not supported')

    return X

def _convert_input_to_torch(
    X: Tensor) -> torch.Tensor:

    if isinstance(X, torch.Tensor):
        pass
    elif isinstance(X, scipy.sparse.spmatrix):
        # TODO: conserve sparseness
        X = torch.tensor(X.toarray())
    elif isinstance(X, np.ndarray):
        X = torch.tensor(X)
    else:
        raise TypeError(f'Input type {type(X)} not supported')

    return X
