"""Relation computation for ParaDime.

The :mod:`paradime.relations` module defines various classes used to compute
relations between data points.
"""

from typing import Union, Callable, Optional
import warnings

import torch
import torch.nn.functional as F
import pynndescent
from scipy.spatial import distance

from paradime import relationdata
from paradime import transforms
from paradime.types import BinaryTensorFun, TensorLike
from paradime import utils

Transform = Union[
    transforms.RelationTransform, list[transforms.RelationTransform]
]


class Relations(utils._ReprMixin):
    """Base class for calculating relations between data points.

    Custom relations should subclass this class.
    """

    def __init__(
        self,
        transform: Optional[Transform] = None,
        data_key: str = "main",
    ):

        self.transform: list[transforms.RelationTransform]
        if transform is None:
            self.transform = []
        elif not isinstance(transform, list):
            self.transform = [transform]
        else:
            self.transform = transform

        self._relations: Optional[relationdata.RelationData] = None

        self.data_key = data_key

    @property
    def relations(self) -> relationdata.RelationData:
        if self._relations is None:
            raise AttributeError(
                "Relations only available after calling 'compute_relations'."
            )
        else:
            return self._relations

    @relations.setter
    def relations(self, reldata: relationdata.RelationData) -> None:
        self._relations = reldata

    def _set_verbosity(self, verbose: bool) -> None:
        if hasattr(self, "verbose"):
            self.verbose = verbose
            for tf in self.transform:
                tf._set_verbosity(verbose)

    def compute_relations(
        self, X: Optional[TensorLike] = None, **kwargs
    ) -> relationdata.RelationData:

        raise NotImplementedError

    def _transform(
        self, X: relationdata.RelationData
    ) -> relationdata.RelationData:

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

    def __init__(
        self,
        X: TensorLike,
        transform: Optional[Transform] = None,
    ):

        super().__init__(transform=transform)

        self._raw_relations = relationdata.relation_factory(X)

    def compute_relations(
        self, X: Optional[TensorLike] = None, **kwargs
    ) -> relationdata.RelationData:
        """Obtain the precomputed relations.

        Args:
            X: Ignored, since relations are already precomputed.

        Returns:
            A :class:`paradime.relationdata.RelationData` instance containing
            the (possibly transformed) relations.
        """

        self.relations = self._transform(self._raw_relations)

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
        data_key: The key to access the data for which to compute relations.
        verbose: Verbosity toggle.

    Attributes:
        relations: A :class:`paradime.relationdata.RelationData` instance
            containing the (possibly transformed) pairwise distances.
            Available only after calling :meth:`compute_relations`.
    """

    def __init__(
        self,
        metric: Optional[Union[Callable, str]] = None,
        transform: Optional[Transform] = None,
        keep_result=True,
        data_key: str = "main",
        verbose: bool = False,
    ):

        if metric is None:
            metric = "euclidean"

        super().__init__(
            transform=transform,
            data_key=data_key,
        )

        self.metric = metric
        self.keep_result = keep_result
        self.verbose = verbose

    def compute_relations(
        self, X: Optional[TensorLike] = None, **kwargs
    ) -> relationdata.RelationData:
        """Calculates the pairwise distances.

        Args:
            X: Input data tensor with one sample per row.

        Returns:
            A :class:`paradime.relationdata.RelationData` instance containing
            the (possibly transformed) pairwise distances.
        """

        if X is None:
            raise ValueError("Missing input for non-precomputed relations.")

        X = utils.convert.to_numpy(X)

        if self._relations is None or not self.keep_result:
            if self.verbose:
                utils.logging.log("Calculating pairwise distances.")
            self.relations = self._transform(
                relationdata.relation_factory(
                    distance.pdist(X, metric=self.metric)
                )
            )
        elif self.verbose:
            utils.logging.log("Using previously calculated distances.")

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
        data_key: The key to access the data for which to compute relations.
        verbose: Verbosity toggle.

    Attributes:
        relations: A :class:`paradime.relationdata.RelationData` instance
            containing the (possibly transformed) pairwise distances.
            Available only after calling :meth:`compute_relations`.
    """

    def __init__(
        self,
        n_neighbors: Optional[int] = None,
        metric: Optional[Union[BinaryTensorFun, str]] = None,
        transform: Optional[Transform] = None,
        data_key: str = "main",
        verbose: bool = False,
    ):

        super().__init__(
            transform=transform,
            data_key=data_key,
        )

        self.n_neighbors = n_neighbors
        self.verbose = verbose
        self.metric = metric

    def _set_n_neighbors(self, num_pts: int) -> None:
        # get highest parameters of any perplexity- or
        # connectivity-based transforms
        perp = 0.0
        n_nb = 0.0
        if self.transform is not None:
            for tf in self.transform:
                if isinstance(tf, transforms.PerplexityBasedRescale):
                    perp = max(perp, tf.perplexity)
                elif isinstance(tf, transforms.ConnectivityBasedRescale):
                    n_nb = max(n_nb, tf.n_neighbors)

        # set number of neighbors according to highest
        # perplexity/n_neighbors found, or to reasonable default
        if self.n_neighbors is None:
            if perp == 0.0 and n_nb == 0.0:
                self.n_neighbors = int(0.05 * num_pts)
            else:
                self.n_neighbors = int(min(num_pts - 1, max(3 * perp, n_nb)))
        else:
            if self.n_neighbors >= 3 * perp and self.n_neighbors >= n_nb:
                self.n_neighbors = min(num_pts - 1, self.n_neighbors)
            elif (
                self.n_neighbors < 3 * perp or self.n_neighbors < n_nb
            ) and 3 * perp > n_nb:
                warnings.warn(
                    f"Number of neighbors {self.n_neighbors} too small for "
                    f"highest perplexity {perp} found in transforms. Using "
                    f"{3 * perp} neighbors (threefold perplexity) instead."
                )
                self.n_neighbors = int(min(num_pts - 1, 3 * perp))
            elif (
                self.n_neighbors < 3 * perp or self.n_neighbors < n_nb
            ) and 3 * perp <= n_nb:
                warnings.warn(
                    f"Number of neighbors {self.n_neighbors} too small for "
                    f"highest 'n_neighbors' {n_nb} found in transforms. "
                    f"Using {n_nb} neighbors instead."
                )
                self.n_neighbors = int(min(num_pts - 1, n_nb))

    def compute_relations(
        self, X: Optional[TensorLike] = None, **kwargs
    ) -> relationdata.RelationData:
        """Calculates the pairwise distances.

        Args:
            X: Input data tensor with one sample per row.

        Returns:
            A :class:`paradime.relationdata.RelationData` instance containing
            the (possibly transformed) pairwise distances.
        """

        if X is None:
            raise ValueError("Missing input for non-precomputed relations.")

        X = utils.convert.to_numpy(X)

        self._set_n_neighbors(X.shape[0])
        assert self.n_neighbors is not None

        if self.verbose:
            utils.logging.log("Indexing nearest neighbors.")

        if self.metric is None:
            self.metric = "euclidean"

        index = pynndescent.NNDescent(
            X, n_neighbors=self.n_neighbors + 1, metric=self.metric
        )
        neighbors, distances = index.neighbor_graph

        self.relations = self._transform(
            relationdata.NeighborRelationTuple((neighbors, distances))
        )

        return self.relations


class DifferentiablePDist(Relations):
    """Differentiable pairwise distances between data points.

    Args:
        p: Parameter that specificies which p-norm to use as
            a distance function. Ignored if ``metric`` is set.
        metric: The distance metric to be used.
        transform: A single :class:`paradime.transforms.Transform` or list of
            :class:`paradime.transforms.Transform` instances to be applied to
            the relations.
        data_key: The key to access the data for which to compute relations.
        verbose: Verbosity toggle.

    Attributes:
        relations: A :class:`paradime.relationdata.RelationData` instance
            containing the (possibly transformed) pairwise distances.
            Available only after calling :meth:`compute_relations`.
    """

    def __init__(
        self,
        p: float = 2,
        metric: Optional[BinaryTensorFun] = None,
        transform: Optional[Transform] = None,
        data_key: str = "main",
    ):

        super().__init__(
            transform=transform,
            data_key=data_key,
        )

        self.metric = metric
        self.metric_p = p

    def compute_relations(
        self, X: Optional[TensorLike] = None, **kwargs
    ) -> relationdata.RelationData:
        """Calculates the pairwise distances.

        If ``metric`` is not None, a flexible but memory-inefficient
        implementation is used instead of PyTorch's
        :func:`torch.nn.functional.pdist`.

        Args:
            X: Input data tensor with one sample per row.

        Returns:
            A :class:`paradime.relationdata.RelationData` instance containing
            the (possibly transformed) pairwise distances.
        """

        if X is None:
            raise ValueError("Missing input for non-precomputed relations.")

        if not isinstance(X, torch.Tensor) or not X.requires_grad:
            warnings.warn(
                "Differentiable pdist operating on tensor "
                "for which no gradients are computed."
            )

        X = utils.convert.to_torch(X)

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
                relationdata.SquareRelationTensor(diss)
            )
        # otherwise use built-in torch method
        else:
            n = X.shape[0]
            diss_cond = F.pdist(X, p=self.metric_p)
            # diss = torch.zeros((n, n), device = X.device)
            # i, j = torch.triu_indices(n, n, offset=1)
            # diss[[i, j]] = diss_cond
            self.relations = self._transform(
                relationdata.TriangularRelationTensor(
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
        data_key: The key to access the data for which to compute relations.

    Attributes:
        relations: A :class:`paradime.relationdata.RelationData` instance
            containing the (possibly transformed) pairwise distances.
            Available only after calling :meth:`compute_relations`.
    """

    def __init__(
        self,
        metric: Optional[BinaryTensorFun] = None,
        transform: Optional[Transform] = None,
        data_key: str = "main",
    ):

        if metric is None:
            metric = lambda a, b: torch.norm(a - b, dim=1)

        super().__init__(
            transform=transform,
            data_key=data_key,
        )

        self.metric = metric

    def compute_relations(
        self, X: Optional[TensorLike] = None, **kwargs
    ) -> relationdata.RelationData:
        """Calculates the distances.

        Args:
            X: Input data tensor of shape (2, n, dim), where n is the number
                of pairs of data points.

        Returns:
            A :class:`paradime.relationdata.RelationData` instance containing
            the (possibly transformed) pairwise distances.
        """

        if X is None:
            raise ValueError("Missing input for non-precomputed relations.")

        X = utils.convert.to_torch(X)

        if len(X) != 2 or X[0].shape != X[1].shape:
            raise ValueError(
                "Expected input tensor of shape (2, n, dim), where n is the "
                "number of pairs of data points."
            )

        self.relations = self._transform(
            relationdata.FlatRelationTensor(self.metric(X[0], X[1]))
        )

        return self.relations
