from datetime import datetime
from multiprocessing.sharedctypes import Value
import warnings
from typing import Union, Callable, Optional
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

Transform = Union[    
    pdtf.RelationTransform,
    list[pdtf.RelationTransform]
]

class Relations():
    """Base class for calculating relations between data points.
    
    Custom relations should subclass this class.
    """
    
    def __init__(self, transform: Optional[Transform] = None):
        
        if transform is None:
            self.transform = transform
        elif not isinstance(transform, list):
            self.transform = [transform]
        else:
            self.transform = transform

        self._relations: Optional[pdreld.RelationData] = None

    @property
    def relations(self) -> pdreld.RelationData:
        if self._relations is None:
            raise AttributeError(
                "Relations only available after calling 'compute_relations'."
            )
        else:
            return self._relations
    
    @relations.setter
    def relations(self, reldata: pdreld.RelationData) -> None:
        self._relations = reldata

    def compute_relations(self,
        X: Optional[Tensor] = None,
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
            :func:`paradime.relationdata.relation_factory`.
        transform: A single :class:`paradime.transforms.Transform` or list of
            :class:`paradime.transforms.Transform` instances to be applied to
            the relations.

    Attributes:
        relations: A :class:`paradime.relationdata.RelationData` instance
        containing the (possibly transformed) relations.
    """

    def __init__(self,
        X: Tensor,
        transform: Optional[Transform] = None,
    ):

        super().__init__(
            transform = transform
        )

        self.relations = self._transform(
            pdreld.relation_factory(X))

    def compute_relations(self,
        X: Optional[Tensor] = None,
        **kwargs
    ) -> pdreld.RelationData:
        """Obtain the precomputed relations.

        Args:
            X: Ignored, since relations are already precomputed.

        Returns:
            A :class:`paradime.relationdata.RelationData` instance containing
            the (possibly transformed) relations.
        """

        if X is not None:
            warnings.warn("Ignoring input for precomputed relations.")
        
        return self.relations


class PDist(Relations):
    """Full pairwise distances between data points.
    
    Args:
        metric: The distance metric to be used.
        transform: A single :class:`paradime.transforms.Transform` or list of
            :class:`paradime.transforms.Transform` instances to be applied to
            the relations.
        keep_result: Specifies whether or not to keep previously
            calculated distances, rather than computing new ones.
        verbose: Verbosity toggle.

    Attributes:
        relations: A :class:`paradime.relationdata.RelationData` instance
            containing the (possibly transformed) pairwise distances.
            Available only after calling :meth:`compute_relations`.
    """
 
    def __init__(self,
        metric: Optional[Metric] = None,
        transform: Optional[Transform] = None,
        keep_result = True,
        verbose: bool = False,
    ):

        if metric is None:
            metric = 'euclidean'

        super().__init__(
            transform=transform
        )

        self.metric = metric
        self.keep_result = keep_result
        self.verbose = verbose

    def compute_relations(self,
        X: Optional[Tensor] = None,
        **kwargs
    ) -> pdreld.RelationData:
        """Calculates the pairwise distances.

        Args:
            X: Input data tensor with one sample per row.

        Returns:
            A :class:`paradime.relationdata.RelationData` instance containing
            the (possibly transformed) pairwise distances.
        """

        if X is None:
            raise ValueError(
                "Missing input for non-precomputed relations."
            )

        X = pdutils._convert_input_to_numpy(X)

        if self._relations is None or not self.keep_result:
            if self.verbose:
                pdutils.report("Calculating pairwise distances.")
            self.relations = self._transform(
                pdreld.relation_factory(pdist(X, metric=self.metric))
            )
        elif self.verbose:
            pdutils.report("Using previously calculated distances.")

        return self.relations


class NeighborBasedPDist(Relations):
    """Approximate, nearest-neighbor-based pairwise distances
    between data points.
    
    Args:
        n_neighbors: Number of nearest neighbors to be considered.
            If not specified, this will be set to 5 percent of the number of
            data points. If the transforms include any
            :class:`paradime.transforms.AdaptiveNeighborhoodRescale` instances,
            this parameter will be overridden according to their parameters.
        metric: The distance metric to be used.
        transform: A single :class:`paradime.transforms.Transform` or list of
            :class:`paradime.transforms.Transform` instances to be applied to
            the relations.
        verbose: Verbosity toggle.

    Attributes:
        relations: A :class:`paradime.relationdata.RelationData` instance
            containing the (possibly transformed) pairwise distances.
            Available only after calling :meth:`compute_relations`.
    """

    def __init__(self,
        n_neighbors: Optional[int] = None,
        metric: Optional[Metric] = None,
        transform: Optional[Transform] = None,
        verbose: bool = False,
    ):

        super().__init__(
            transform=transform
        )

        self.n_neighbors = n_neighbors
        self.verbose = verbose
        self.metric = metric

    def _set_n_neighbors(self, num_pts: int) -> None:
        # get highest parameters of any perplexity- or
        # connectivity-based transforms
        perp = 0.
        n_nb = 0.
        if self.transform is not None:
            for tf in self.transform:
                if isinstance(tf, pdtf.PerplexityBasedRescale):
                    perp = max(perp, tf.perplexity)
                elif isinstance(tf, pdtf.ConnectivityBasedRescale):
                    n_nb = max(n_nb, tf.n_neighbors)

        # set number of neighbors according to highest
        # perplexity/n_neighbors found, or to reasonable default
        if self.n_neighbors is None:
            if perp == 0. and n_nb == 0.:
                self.n_neighbors = int(0.05 * num_pts)
            else:
                self.n_neighbors = int(min(num_pts - 1, max(3 * perp, n_nb)))
        else:
            if self.n_neighbors >= 3 * perp and self.n_neighbors >= n_nb:
                self.n_neighbors = min(num_pts - 1, self.n_neighbors)
            elif ((self.n_neighbors < 3 * perp or self.n_neighbors < n_nb) and
                3 * perp > n_nb):
                warnings.warn(
                    f"Number of neighbors {self.n_neighbors} too small for "
                    f"highest perplexity {perp} found in transforms. Using "
                    f"{3 * perp} neighbors (threefold perplexity) instead."
                )
                self.n_neighbors = int(min(num_pts - 1, 3 * perp))
            elif ((self.n_neighbors < 3 * perp or self.n_neighbors < n_nb) and
                3 * perp <= n_nb):
                warnings.warn(
                    f"Number of neighbors {self.n_neighbors} too small for "
                    f"highest 'n_neighbors' {n_nb} found in transforms. "
                    f"Using {n_nb} neighbors instead."
                )
                self.n_neighbors = int(min(num_pts - 1, n_nb))
    
    def compute_relations(self,
        X: Optional[Tensor] = None,
        **kwargs
    ) -> pdreld.RelationData:
        """Calculates the pairwise distances.

        Args:
            X: Input data tensor with one sample per row.

        Returns:
            A :class:`paradime.relationdata.RelationData` instance containing
            the (possibly transformed) pairwise distances.
        """

        if X is None:
            raise ValueError(
                "Missing input for non-precomputed relations."
            )

        X = pdutils._convert_input_to_numpy(X)

        self._set_n_neighbors(X.shape[0])
        assert self.n_neighbors is not None
        
        if self.verbose:
            pdutils.report("Indexing nearest neighbors.")

        if self.metric is None:
            self.metric = 'euclidean'
        
        index = NNDescent(X,
            n_neighbors=self.n_neighbors + 1,
            metric=self.metric
        )
        neighbors, distances = index.neighbor_graph

        self.relations = self._transform(
            pdreld.NeighborRelationTuple(
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
        transform: A single :class:`paradime.transforms.Transform` or list of
            :class:`paradime.transforms.Transform` instances to be applied to
            the relations.
        verbose: Verbosity toggle.

    Attributes:
        relations: A :class:`paradime.relationdata.RelationData` instance
            containing the (possibly transformed) pairwise distances.
            Available only after calling :meth:`compute_relations`.
    """

    def __init__(self,
        p: float = 2,
        metric: Optional[Callable[
            [torch.Tensor, torch.Tensor],
            torch.Tensor]] = None,
        transform: Optional[Transform] = None,
    ):

        super().__init__(
            transform=transform
        )

        self.metric = metric
        self.metric_p = p

    def compute_relations(self,
        X: Optional[Tensor] = None,
        **kwargs
    ) -> pdreld.RelationData:
        """Calculates the pairwise distances.
        
        If `metric` is not None, a flexible but memory-inefficient
        implementation is used instead of PyTorch's
        :func:`torch.nn.functional.pdist`.

        Args:
            X: Input data tensor with one sample per row.

        Returns:
            A :class:`paradime.relationdata.RelationData` instance containing
            the (possibly transformed) pairwise distances.
        """

        if X is None:
            raise ValueError(
                "Missing input for non-precomputed relations."
            )

        if not isinstance(X, torch.Tensor) or not X.requires_grad:
            warnings.warn(
                "Differentiable pdist operating on tensor "
                "for which no gradients are computed."
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
            # diss = torch.zeros((n, n), device = X.device)
            # i, j = torch.triu_indices(n, n, offset=1)
            # diss[[i, j]] = diss_cond
            self.relations = self._transform(
                pdreld.TriangularRelationTensor(
                    diss_cond
                    # diss + diss.T
                )
            )

        return self.relations

class DistsFromTo(Relations):
    """Distances between individual pairs of data points.
    
    Args:
        metric: The distance metric to be used.
        transform: A single :class:`paradime.transforms.Transform` or list of
            :class:`paradime.transforms.Transform` instances to be applied to
            the relations.

    Attributes:
        relations: A :class:`paradime.relationdata.RelationData` instance
            containing the (possibly transformed) pairwise distances.
            Available only after calling :meth:`compute_relations`.
    """
 
    def __init__(self,
        metric: Optional[Callable[
            [torch.Tensor, torch.Tensor],
            torch.Tensor]] = None,
        transform: Optional[Transform] = None,
    ):

        if metric is None:
            metric = (lambda a, b: torch.norm(a - b, dim=1))

        super().__init__(
            transform=transform
        )

        self.metric = metric

    def compute_relations(self,
        X: Optional[Tensor] = None,
        **kwargs
    ) -> pdreld.RelationData:
        """Calculates the distances.

        Args:
            X: Input data tensor of shape (2, n, dim), where n is the number
                of pairs of data points.

        Returns:
            A :class:`paradime.relationdata.RelationData` instance containing
            the (possibly transformed) pairwise distances.
        """

        if X is None:
            raise ValueError(
                "Missing input for non-precomputed relations."
            )

        X = pdutils._convert_input_to_torch(X)

        if len(X) != 2 or X[0].shape != X[1].shape:
            raise ValueError(
                "Expected input tensor of shape (2, n, dim), where n is the "
                "number of pairs of data points."
            )

        self.relations = self._transform(
            pdreld.FlatRelationTensor(self.metric(X[0], X[1]))
        )

        return self.relations