from datetime import datetime
import warnings
from typing import Union, Callable, List, Literal, Tuple, Any
import torch
import torch.nn.functional as F
import numpy as np
from pynndescent import NNDescent
import scipy.sparse
from scipy.spatial.distance import pdist, squareform

from paradime import affinitydata as prdmad
from paradime import transforms as prdmtf
from .types import Metric, Tensor, Diss
from .utils import report

SingleTransform = Union[
        Callable[
            [prdmad.AffinityData],
            prdmad.AffinityData
        ],
        prdmtf.AffinityTransform
    ]

Transform = Union[    
    SingleTransform,
    List[SingleTransform]
]

class Affinity():
    
    def __init__(self,
        transform: Transform = None
        ) -> None:
        
        if transform is None:
            self.transform = transform
        elif not isinstance(transform, list):
            self.transform = [transform]
        else:
            self.transform = transform

    def compute_affinities(self,
        X: Tensor = None,
        **kwargs) -> prdmad.AffinityData:

        raise NotImplementedError

    def _transform(self,
        X: prdmad.AffinityData
        ) -> prdmad.AffinityData:

        if self.transform is None:
            return X
        else:
            for tf in self.transform:
                X = tf(X)
            return X


class Precomputed(Affinity):

    def __init__(self,
        X: Tensor,
        transform: Union[Callable, prdmtf.AffinityTransform] = None
        ) -> None:

        super().__init__(
            transform = transform
        )

        self.affinities = self._transform(
            prdmad.affinity_factory(X))

    def compute_affinities(self,
        X: Tensor = None,
        **kwargs
        ) -> prdmad.AffinityData:

        if X is not None:
            warnings.warn('Ignoring input for precomputed affinity.')
        
        return self.affinities


class Exact(Affinity):
 
    def __init__(self,
        metric: Metric = None,
        keep_result = True,
        transform: Union[Callable, prdmtf.AffinityTransform] = None,
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

    def compute_affinities(self,
        X: Tensor = None,
        **kwargs
        ) -> prdmad.AffinityData:

        if X is None:
            raise ValueError(
                'Missing input for non-precomputed affinity.'
            )

        X = _convert_input_to_numpy(X)

        if not hasattr(self, 'affinities') or not self.keep_result:
            if self.verbose:
                report('Calculating pairwise distances.')
            self.affinities = self._transform(
                prdmad.affinity_factory(pdist(X, metric=self.metric))
            )
        elif self.verbose:
            report('Using previously calculated distances.')

        return self.affinities


class NeighborBased(Affinity):

    def __init__(self,
        n_neighbors: int = None,
        metric: Metric = None,
        transform: Union[Callable, prdmtf.AffinityTransform] = None,
        verbose: Union[bool, int] = False
        ) -> None:

        super().__init__(
            transform=transform
        )

        self.n_neighbors = n_neighbors
        self.verbose = verbose
        self.metric = metric
    
    def compute_affinities(self,
        X: Tensor = None,
        **kwargs
        ) -> prdmad.AffinityData:

        if X is None:
            raise ValueError(
                'Missing input for non-precomputed affinity.'
            )

        X = _convert_input_to_numpy(X)

        num_pts = X.shape[0]

        # get highest perplexity of any PerplexityBased transforms
        perp = 0.
        if self.transform is not None:
            for tf in self.transform:
                if isinstance(tf, prdmtf.PerplexityBased):
                    perp = max(perp, tf.perplexity)

        # set number of neighbors according to highest
        # perplexity found, or to reasonable default
        if self.n_neighbors is None:
            if perp == 0:
                self.n_neighbors = int(0.05 * num_pts)
            else:
                self.n_neighbors = int(min(num_pts - 1, 3 * perp))
        else:
            if self.n_neighbors > 3 * perp:
                self.n_neighbors = min(num_pts - 1, self.n_neighbors)
            elif self.n_neighbors < 3 * perp:
                warnings.warn(
                    f'Number of neighbors {self.n_neighbors} too small.' +
                    f'Using 3 * perplexity {perp} = {3 * perp} instead.'
                )
                self.n_neighbors = int(min(num_pts - 1, 3 * perp))
        
        if self.verbose:
            report('Indexing nearest neighbors.')

        if self.metric is None:
            self.metric = 'euclidean'
        
        index = NNDescent(X,
            n_neighbors=self.n_neighbors,
            metric = self.metric
        )
        neighbors, distances = index.neighbor_graph

        self.affinities = self._transform(
            prdmad.affinity_factory(
                (neighbors, distances)
            )
        )

        return self.affinities


class Differentiable(Affinity):

    def __init__(self,
        p: float = 2,
        metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
        transform: Union[Callable, prdmtf.AffinityTransform] = None,
        verbose: Union[int, bool] = False
        ) -> None:

        super().__init__(
            transform=transform
        )

        self.metric = metric
        self.metric_p = p

        self.verbose = verbose

    def compute_affinities(self,
        X: Tensor = None,
        **kwargs
        ) -> prdmad.AffinityData:

        if X is None:
            raise ValueError(
                'Missing input for non-precomputed affinity.'
            )

        if not isinstance(X, torch.Tensor) or not X.requires_grad:
            warnings.warn(
                'Differentiable affinity operating on tensor ' +
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
            self.affinities = self._transform(
                prdmad.SquareAffinityTensor(diss)
            )
        # otherwise use built-in torch method
        else:
            n = X.shape[0]
            diss_cond = F.pdist(X, p=self.metric_p)
            diss = torch.zeros((n, n), device = X.device)
            i, j = torch.triu_indices(n, n, offset=1)
            diss[[i, j]] = diss_cond
            self.affinities = self._transform(
                prdmad.SquareAffinityTensor(
                    diss + diss.T
                )
            )

        return self.affinities



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
