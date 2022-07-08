from datetime import datetime
import warnings
from typing import Union, Callable, List, Literal, Tuple, Any
import torch
import torch.nn.functional as F
import numpy as np
from pynndescent import NNDescent
import scipy.sparse
from scipy.spatial.distance import pdist, squareform

import paradime.relationdata as pdreld
import paradime.transforms as pdtf
import paradime.utils as pdutils

from .types import Metric, Tensor, Rels

SingleTransform = Union[
        Callable[
            [pdreld.RelationData],
            pdreld.RelationData
        ],
        pdtf.RelationTransform
    ]

Transform = Union[    
    SingleTransform,
    List[SingleTransform]
]

class Relations():
    """Base class for calculating relations between data points.
    
    Custom relations should subclass this class.
    """
    
    def __init__(self,
        transform: Transform = None
        ) -> None:
        
        if transform is None:
            self.transform = transform
        elif not isinstance(transform, list):
            self.transform = [transform]
        else:
            self.transform = transform

    def compute_relations(self,
        X: Tensor = None,
        **kwargs) -> pdreld.RelationData:

        raise NotImplementedError

    def _transform(self,
        X: pdreld.RelationData
        ) -> pdreld.RelationData:

        if self.transform is None:
            return X
        else:
            for tf in self.transform:
                X = tf(X)
            return X


class Precomputed(Relations):
    """Precomputed relations between data points.

    Args:
        X: The precomputed relations, in a form accepted by
            :func:`relation_factory`.
        transform: A single transform or list of transforms
            to be applied to the relations.

    Attributes:
        relations: A :class:`RelationData` instance containing the
        (possibly transformed) relations.
    """

    def __init__(self,
        X: Tensor,
        transform: Union[Callable, pdtf.RelationTransform] = None
        ) -> None:

        super().__init__(
            transform = transform
        )

        self.relations = self._transform(
            pdreld.relation_factory(X))

    def compute_relations(self,
        X: Tensor = None,
        **kwargs
        ) -> pdreld.RelationData:
        """Obtain the precomputed relations.

        Args:
            X: Ignored, since relations are already precomputed.

        Returns:
            A RelationData instance containing the (possibly
            transformed) relations.
        """

        if X is not None:
            warnings.warn('Ignoring input for precomputed relations.')
        
        return self.relations


class PDist(Relations):
    """Full pairwise distances between data points.
    
    Args:
        metric: The distance metric to be used.
        transform: A single transform or list of transforms
            to be applied to the relations.
        keep_result: Specifies whether or not to keep previously
            calculated distances, rather than computing new ones.
        verbose: Verbosity toggle.

    Attributes:
        relations: A RelationData instance containing the (possibly
        transformed) pairwise distances. Available only after
        calling :meth:`compute_relations`.
    """
 
    def __init__(self,
        metric: Metric = None,
        transform: Union[Callable, pdtf.RelationTransform] = None,
        keep_result = True,
        verbose: bool = False
        ) -> None:

        if metric is None:
            metric = 'euclidean'

        super().__init__(
            transform=transform
        )

        self.metric = metric
        self.keep_result = keep_result
        self.verbose = verbose

    def compute_relations(self,
        X: Tensor = None,
        **kwargs
        ) -> pdreld.RelationData:
        """Calculates the pairwise distances.

        Args:
            X: Input data tensor with one sample per row.

        Returns:
            A RelationData instance containing the (possibly
            transformed) pairwise distances.
        """

        if X is None:
            raise ValueError(
                'Missing input for non-precomputed relations.'
            )

        X = pdutils._convert_input_to_numpy(X)

        if not hasattr(self, 'relations') or not self.keep_result:
            if self.verbose:
                pdutils.report('Calculating pairwise distances.')
            self.relations = self._transform(
                pdreld.relation_factory(pdist(X, metric=self.metric))
            )
        elif self.verbose:
            pdutils.report('Using previously calculated distances.')

        return self.relations


class NeighborBasedPDist(Relations):
    """Approximate, nearest-neighbor-based pairwise distances
    between data points.
    
    Args:
        n_neighbors: Number of nearest neighbors to be considered.
            If not specified, this will be set to 5 percent of the
            data points. If the transforms include any perplexity-based
            ones, this parameter will be overridden according to the
            highest perplexity.
        metric: The distance metric to be used.
        transform: A single transform or list of transforms
            to be applied to the relations.
        verbose: Verbosity toggle.

    Attributes:
        relations: A RelationData instance containing the (possibly
        transformed) pairwise distances. Available only after
        calling :meth:`compute_relations`.
    """

    def __init__(self,
        n_neighbors: int = None,
        metric: Metric = None,
        transform: Union[Callable, pdtf.RelationTransform] = None,
        verbose: bool = False
        ) -> None:

        super().__init__(
            transform=transform
        )

        self.n_neighbors = n_neighbors
        self.verbose = verbose
        self.metric = metric
    
    def compute_relations(self,
        X: Tensor = None,
        **kwargs
        ) -> pdreld.RelationData:
        """Calculates the pairwise distances.

        Args:
            X: Input data tensor with one sample per row.

        Returns:
            A RelationData instance containing the (possibly
            transformed) pairwise distances.
        """

        if X is None:
            raise ValueError(
                'Missing input for non-precomputed relations.'
            )

        X = pdutils._convert_input_to_numpy(X)

        num_pts = X.shape[0]

        # get highest perplexity of any PerplexityBased transforms
        perp = 0.
        if self.transform is not None:
            for tf in self.transform:
                if isinstance(tf, pdtf.PerplexityBased):
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
            pdutils.report('Indexing nearest neighbors.')

        if self.metric is None:
            self.metric = 'euclidean'
        
        index = NNDescent(X,
            n_neighbors=self.n_neighbors + 1,
            metric = self.metric
        )
        neighbors, distances = index.neighbor_graph

        self.relations = self._transform(
            pdreld.relation_factory(
                (neighbors, distances)
            )
        )

        return self.relations


class DifferentiablePDist(Relations):
    """Differentiable pairwise distances between data points.
    
    Args:
        p: Parameter that specificies which p-norm to use as
            a distance function. Ignored if `metric` is set.
        metric: The distance metric to be used.
        transform: A single transform or list of transforms
            to be applied to the relations.
        verbose: Verbosity toggle.

    Attributes:
        relations: A RelationData instance containing the (possibly
        transformed) pairwise distances. Available only after
        calling :meth:`compute_relations`.
    """

    def __init__(self,
        p: float = 2,
        metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
        transform: Union[Callable, pdtf.RelationTransform] = None,
        ) -> None:

        super().__init__(
            transform=transform
        )

        self.metric = metric
        self.metric_p = p

    def compute_relations(self,
        X: Tensor = None,
        **kwargs
        ) -> pdreld.RelationData:
        """Calculates the pairwise distances. If :param:`metric` is
        not None, flexible memory-inefficient implementation
        is used instead of PyTorch's `pdist`.

        Args:
            X: Input data tensor with one sample per row.

        Returns:
            A RelationData instance containing the (possibly
            transformed) pairwise distances.
        """

        if X is None:
            raise ValueError(
                'Missing input for non-precomputed relations.'
            )

        if not isinstance(X, torch.Tensor) or not X.requires_grad:
            warnings.warn(
                'Differentiable pdist operating on tensor ' +
                'for which no gradients are computed.'
            )

        X = pdutils._convert_input_to_torch(X)

        # use memory-inefficient pdist to allow for arbitrary metrics
        # will break for large batches
        if self.metric is not None:
            n = X.shape[0]
            expanded = X.unsqueeze(1)
            # repeat entries n times
            tiled = torch.repeat_interleave(expanded, n, dim=1)
            # apply metric to pairs of items
            diss = self.metric(tiled, tiled.transpose(0, 1))
            self.relations = self._transform(
                pdreld.SquareRelationTensor(diss)
            )
        # otherwise use built-in torch method
        else:
            n = X.shape[0]
            diss_cond = F.pdist(X, p=self.metric_p)
            diss = torch.zeros((n, n), device = X.device)
            i, j = torch.triu_indices(n, n, offset=1)
            diss[[i, j]] = diss_cond
            self.relations = self._transform(
                pdreld.SquareRelationTensor(
                    diss + diss.T
                )
            )

        return self.relations
