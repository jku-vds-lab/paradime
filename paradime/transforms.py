"""Relation transforms for ParaDime.

The :mod:`paradime.tranforms` module defines various classes used to transform
relations between data points.
"""

import functools
from typing import Callable, Union, Optional, Any

import numba
import numpy as np
import torch
import scipy.optimize
import scipy.sparse

from paradime import relationdata
from paradime import utils


class RelationTransform(utils._ReprMixin):
    """Base class for relation transforms.

    Custom transforms should subclass this class.
    """

    def __init__(self, **kwargs):
        pass

    def __call__(
        self,
        reldata: relationdata.RelationData,
    ) -> relationdata.RelationData:

        return self.transform(reldata)

    def transform(
        self,
        reldata: relationdata.RelationData,
    ) -> relationdata.RelationData:
        """Applies the transform to input data.

        Args:
            reldata: The :class:`paradime.relationdata.RelationData` instance
                to be transformed.

        Returns:
            A :class:`paradime.relationdata.RelationData` instance containing
                the transformed relation values.
        """

        raise NotImplementedError()

    def _set_verbosity(self, verbose: bool) -> None:
        if hasattr(self, "verbose"):
            self.verbose = verbose


class Identity(RelationTransform):
    """A placeholder identity transform."""

    def transform(
        self,
        reldata: relationdata.RelationData,
    ) -> relationdata.RelationData:
        return reldata


class ToFlatArray(RelationTransform):
    """Converts the relations to a
    :class:`paradime.relationdata.FlatRelationArray`.
    """

    def transform(
        self,
        reldata: relationdata.RelationData,
    ) -> relationdata.RelationData:
        return reldata.to_flat_array()


class ToFlatTensor(RelationTransform):
    """Converts the relations to a
    :class:`paradime.relationdata.FlatRelationTensor`.
    """

    def transform(
        self,
        reldata: relationdata.RelationData,
    ) -> relationdata.RelationData:
        return reldata.to_flat_tensor()


class ToSquareArray(RelationTransform):
    """Converts the relations to a
    :class:`paradime.relationdata.SquareRelationArray`.
    """

    def transform(
        self,
        reldata: relationdata.RelationData,
    ) -> relationdata.RelationData:
        return reldata.to_square_array()


class ToSquareTensor(RelationTransform):
    """Converts the relations to a
    :class:`paradime.relationdata.SquareRelationTensor`.
    """

    def transform(
        self,
        reldata: relationdata.RelationData,
    ) -> relationdata.RelationData:
        return reldata.to_square_tensor()


class ToTriangularArray(RelationTransform):
    """Converts the relations to a
    :class:`paradime.relationdata.TriangularRelationArray`.
    """

    def transform(
        self,
        reldata: relationdata.RelationData,
    ) -> relationdata.RelationData:
        return reldata.to_triangular_array()


class ToTriangularTensor(RelationTransform):
    """Converts the relations to a
    :class:`paradime.relationdata.TriangularRelationTensor`.
    """

    def transform(
        self,
        reldata: relationdata.RelationData,
    ) -> relationdata.RelationData:
        return reldata.to_triangular_tensor()


class ToNeighborTuple(RelationTransform):
    """Converts the relations to a
    :class:`paradime.relationdata.NeighborRelationTuple`.
    """

    def transform(
        self,
        reldata: relationdata.RelationData,
    ) -> relationdata.RelationData:
        return reldata.to_neighbor_tuple()


class ToSparseArray(RelationTransform):
    """Converts the relations to a
    :class:`paradime.relationdata.SparseRelationArray`.
    """

    def transform(
        self,
        reldata: relationdata.RelationData,
    ) -> relationdata.RelationData:
        return reldata.to_sparse_array()


class AdaptiveNeighborhoodRescale(RelationTransform):
    """Rescales relation values for each data point based on its neighbors.

    This is a base class for transformations such as those used by t-SNE
    or UMAP. For each data point, a parameter is fitted by comparing
    kernel-transformed relations to a target value. Once the parameter
    value is found, the kernel function is used to transform the relations.

    Args:
        kernel: The kernel function used to transform the relations. This
            is a callable taking the relation values for a data point,
            along with a parameter.
        find_param: The function used to find the parameter value. This is a
            callable taking the relation values and a fixed value to compare
            the transformed relations against.
        verbose: Verbosity toggle.
    """

    def __init__(
        self,
        kernel: Callable[[np.ndarray, float], Union[float, np.ndarray]],
        find_param: Callable[..., float],
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.kernel = kernel
        self.find_param = find_param
        self.verbose = verbose
        self.kwargs = kwargs

        self._param_values: Optional[np.ndarray] = None

    @property
    def param_values(self) -> np.ndarray:
        """The parameter values determined for each data point. Available only
        after calling the transform.
        """
        if self._param_values is not None:
            return self._param_values
        else:
            raise AttributeError(
                "Parameter values only available after calling transform."
            )

    def _set_root_scalar_defaults(self) -> None:
        raise NotImplementedError()

    def _get_comparison_constant(self) -> float:
        raise NotImplementedError()

    def transform(
        self,
        reldata: relationdata.RelationData,
    ) -> relationdata.RelationData:

        reldata = reldata.to_neighbor_tuple()
        reldata._remove_self_relations()

        self._set_root_scalar_defaults()

        neighbors, relations = reldata.data
        num_pts = len(neighbors)
        self._param_values = np.empty(num_pts, dtype=float)

        if self.verbose:
            utils.logging.log("Calculating probabilities.")

        for i, rels in enumerate(relations):
            beta = self.find_param(
                rels, self._get_comparison_constant(), **self.kwargs
            )
            self._param_values[i] = beta
            relations[i] = self.kernel(rels, beta)

        reldata.data = (neighbors, relations)

        return reldata


class PerplexityBasedRescale(AdaptiveNeighborhoodRescale):
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

    def __init__(
        self,
        perplexity: float = 30,
        verbose: bool = False,
        **kwargs,  # passed on to root_scalar
    ):
        super().__init__(_p_i, _find_beta, verbose=verbose, **kwargs)

        self.perplexity = perplexity

    @property
    def betas(self) -> np.ndarray:
        return self.param_values

    def _get_comparison_constant(self) -> float:
        return self.perplexity

    def _set_root_scalar_defaults(self) -> None:
        if not self.kwargs:  # check if emtpy
            self.kwargs["bracket"] = [0.01, 1.0]


@numba.jit
def _entropy(dists: np.ndarray, beta: float) -> float:
    x = -(dists**2) * beta
    y = np.exp(x)
    ysum = y.sum()

    if ysum < 1e-50:
        result = -1.0
    else:
        factor = -1 / (np.log(2.0) * ysum)
        result = factor * ((y * x) - (y * np.log(ysum))).sum()

    return result


def _p_i(dists: np.ndarray, beta: float) -> np.ndarray:
    x = -(dists**2) * beta
    y = np.exp(x)
    ysum = y.sum()

    return y / ysum


def _find_beta(dists: np.ndarray, perp: float, **kwargs) -> float:
    return scipy.optimize.root_scalar(
        lambda b: _entropy(dists, b) - np.log2(perp), **kwargs
    ).root


class ConnectivityBasedRescale(AdaptiveNeighborhoodRescale):
    """Applies a connectivity-based transformation to the relation values.

    The relation values are rescaled using shifted Guassian kernels. The
    shift is equal to the closes neighboring data point, and the kernel
    width is set by by comparing the summed kernel values to the binary
    logarithm of the specified number of neighbors. This is the relation
    transform used by UMAP.

    Args:
        n_neighbors: The number of nearest neighbors used to determine the
            kernel widths.
        verbose: Verbosity toggle.
        **kwargs: Passed on to :func:`scipy.optimize.root_scalar`, which
            determines the kernel widths. By default, this is set to use a
            bracket of [10^(-6), 10^6] for the root search.
    """

    def __init__(
        self,
        n_neighbors: float = 15,
        verbose: bool = False,
        **kwargs,  # passed on to root_scalar
    ):
        super().__init__(_exp_k, _find_sigma, verbose=verbose, **kwargs)

        self.n_neighbors = n_neighbors

    @property
    def sigmas(self) -> np.ndarray:
        return self.param_values

    def _get_comparison_constant(self) -> float:
        return self.n_neighbors

    def _set_root_scalar_defaults(self) -> None:
        if not self.kwargs:  # check if emtpy
            self.kwargs["bracket"] = [1.0e-6, 1.0e6]


def _exp_k(dists: np.ndarray, sigma: float) -> np.ndarray:
    x = dists - dists.min()
    return np.exp(-x / sigma)


def _find_sigma(dists: np.ndarray, k: int, **kwargs) -> float:
    return scipy.optimize.root_scalar(
        lambda s: _exp_k(dists, s).sum() - np.log2(float(k)), **kwargs
    ).root


class Symmetrize(RelationTransform):
    """Symmetrizes the relation values.

    Args:
        subtract_product: Specifies which symmetrization routine to use.
            If set to False (default), a matrix M is symmetrized by
            calculating 1/2 * (M + M^T); if set to True, M is symmetrized
            by calculating M + M^T - M * M^T, where '*' is the element-wise
            (Hadamard) product.
    """

    def __init__(self, subtract_product: bool = False):
        super().__init__()
        self.subtract_product = subtract_product

    def transform(
        self,
        reldata: relationdata.RelationData,
    ) -> relationdata.RelationData:

        if isinstance(
            reldata,
            (
                relationdata.TriangularRelationArray,
                relationdata.TriangularRelationTensor,
            ),
        ):
            return reldata
        else:
            if self.subtract_product:
                symmetrizer = _sym_subtract_product
            else:
                symmetrizer = _sym_plus_only

            if isinstance(reldata, relationdata.NeighborRelationTuple):
                return symmetrizer(reldata.to_sparse_array())
            else:
                return symmetrizer(reldata)


def _sym_plus_only(
    reldata: relationdata.RelationData,
) -> relationdata.RelationData:
    if isinstance(reldata, relationdata.SquareRelationArray):
        reldata.data = 0.5 * (reldata.data + reldata.data.T)
    elif isinstance(reldata, relationdata.SparseRelationArray):
        reldata.data = 0.5 * (reldata.data + reldata.data.transpose())
    elif isinstance(reldata, relationdata.SquareRelationTensor):
        reldata.data = 0.5 * (reldata.data + torch.t(reldata.data))
    else:
        raise TypeError(
            "Expected non-flat :class:`paradime.relationdata.RelationData`."
        )
    return reldata


def _sym_subtract_product(
    reldata: relationdata.RelationData,
) -> relationdata.RelationData:
    if isinstance(reldata, relationdata.SquareRelationArray):
        reldata.data = (
            reldata.data + reldata.data.T - reldata.data * reldata.data.T
        )
    elif isinstance(reldata, relationdata.SparseRelationArray):
        reldata.data = (
            reldata.data
            + reldata.data.transpose()
            - reldata.data.multiply(reldata.data.transpose())
        )
    elif isinstance(reldata, relationdata.SquareRelationTensor):
        reldata.data = (
            reldata.data
            + torch.t(reldata.data)
            - reldata.data * torch.t(reldata.data)
        )
    else:
        raise TypeError(
            "Expected non-flat :class:`paradime.relationdata.RelationData`."
        )
    return reldata


class NormalizeRows(RelationTransform):
    """Normalizes the relation values for each data point separately."""

    def transform(
        self,
        reldata: relationdata.RelationData,
    ) -> relationdata.RelationData:

        if isinstance(reldata, relationdata.TriangularRelationArray):
            reldata = reldata.to_square_array()
            reldata.data /= reldata.data.sum(axis=1, keepdims=True)
        elif isinstance(reldata, relationdata.TriangularRelationTensor):
            reldata = reldata.to_square_tensor()
            reldata.data /= reldata.data.sum(dim=1, keepdim=True)
        elif isinstance(reldata, relationdata.SquareRelationArray):
            reldata.data /= reldata.data.sum(axis=1, keepdims=True)
        elif isinstance(reldata, relationdata.SquareRelationTensor):
            reldata.data /= reldata.data.sum(dim=1, keepdim=True)
        elif isinstance(reldata, relationdata.SparseRelationArray):
            norm_factors = 1 / np.repeat(
                np.array(reldata.data.sum(axis=1)), reldata.data.getnnz(axis=1)
            )
            reldata.data = scipy.sparse.csr_matrix(
                (norm_factors, reldata.data.nonzero())
            ).multiply(reldata.data)
        elif isinstance(reldata, relationdata.NeighborRelationTuple):
            neighbors, relations = reldata.data
            reldata.data = (
                neighbors,
                relations / relations.sum(axis=1, keepdims=True),
            )
        else:
            raise TypeError(
                "Expected non-flat :class:`paradime.relationdata.RelationData`."
            )
        return reldata


class Normalize(RelationTransform):
    """Normalizes all relations at once."""

    def transform(
        self,
        reldata: relationdata.RelationData,
    ) -> relationdata.RelationData:

        if isinstance(
            reldata,
            (
                relationdata.TriangularRelationArray,
                relationdata.TriangularRelationTensor,
            ),
        ):
            reldata.data /= 2 * reldata.data.sum()
        elif isinstance(
            reldata,
            (
                relationdata.SquareRelationArray,
                relationdata.SquareRelationTensor,
                relationdata.SparseRelationArray,
                relationdata.FlatRelationTensor,
                relationdata.FlatRelationArray,
            ),
        ):
            reldata.data /= reldata.data.sum()
        elif isinstance(reldata, relationdata.NeighborRelationTuple):
            neighbors, relations = reldata.data
            reldata.data = (neighbors, relations / relations.sum())
        else:
            raise TypeError(
                "Expected :class:`paradime.relationdata.RelationData`."
            )
        return reldata


class ZeroDiagonal(RelationTransform):
    """Sets all self-relations to zero."""

    def transform(
        self,
        reldata: relationdata.RelationData,
    ) -> relationdata.RelationData:

        if isinstance(
            reldata,
            (
                relationdata.TriangularRelationArray,
                relationdata.TriangularRelationTensor,
            ),
        ):
            return reldata
        elif isinstance(reldata, relationdata.SquareRelationArray):
            np.fill_diagonal(reldata.data, 0.0)
        elif isinstance(reldata, relationdata.SquareRelationTensor):
            reldata.data.fill_diagonal_(0.0)
        elif isinstance(reldata, relationdata.SparseRelationArray):
            reldata.data.setdiag(0.0)
        elif isinstance(reldata, relationdata.NeighborRelationTuple):
            reldata._remove_self_relations()
        else:
            raise TypeError(
                "Expected non-flat :class:`paradime.relationdata.RelationData`."
            )
        return reldata


class StudentTTransform(RelationTransform):
    """Transforms relations based on Student's t-distribution.

    Args:
        alpha: Degrees of freedom of the distribution. This can either be
            a float or a PyTorch tensor. Alpha can be optimized together
            with the DR model in a :class:`paradime.dr.ParametricDR` by
            setting it to one of the model's additional parameters.
    """

    def __init__(self, alpha: Union[float, torch.Tensor]):
        self.alpha = alpha

    def transform(
        self,
        reldata: relationdata.RelationData,
    ) -> relationdata.RelationData:

        if isinstance(
            reldata,
            (
                relationdata.SquareRelationTensor,
                relationdata.TriangularRelationTensor,
                relationdata.FlatRelationTensor,
            ),
        ):
            reldata.data = reldata.data.pow(2.0)
            reldata.data = reldata.data / self.alpha
            reldata.data = reldata.data + 1.0
            # TODO: fix RuntimeError due to in-place operation below
            reldata.data = reldata.data.pow(-(self.alpha + 1.0) / 2.0)
        elif isinstance(
            reldata,
            (
                relationdata.SquareRelationArray,
                relationdata.TriangularRelationArray,
                relationdata.FlatRelationArray,
            ),
        ):
            reldata.data = np.power(
                1.0 + reldata.data**2.0 / self.alpha,
                -(self.alpha + 1.0) / 2.0,
            )
        elif isinstance(reldata, relationdata.SparseRelationArray):
            return self.transform(reldata.to_square_array())
        elif isinstance(reldata, relationdata.NeighborRelationTuple):
            neighbors, rels = reldata.data
            rels = np.power(
                1.0 + rels**2.0 / self.alpha, -(self.alpha + 1.0) / 2.0
            )
            reldata.data = (neighbors, rels)
        else:
            raise TypeError(
                "Expected :class:`paradime.relationdata.RelationData`."
            )
        return reldata


class ModifiedCauchyTransform(RelationTransform):
    """Transforms relations based on a modified Cauchy distribution.

    This transform applies a modified Cauchy distribution function
    to the relations. The distribution's parameters ``a`` and ``b`` are
    determined from the parameters ``min_dist`` and ``spread`` by fitting a
    smooth approximation of an offset exponential decay.

    Args:
        min_dist: Effective minimum distance of points if the transformed
            relations were to be used for calculating an embedding.
        spread: Effective scale of the points if the tranformed relations
            were to be used for calculating an embedding.
        a: Parameter to define the distribution directly. It can be optimized
            together with the DR model in a :class:`ParametricDR` by setting
            it to one of the model's additional parameters.
        b: Parameter to define the distribution directly. It can be optimized
            together with the DR model in a :class:`ParametricDR` by setting
            it to one of the model's additional parameters.
    """

    def __init__(
        self,
        min_dist: float = 0.1,
        spread: float = 1.0,
        a: Optional[Union[float, torch.Tensor]] = None,
        b: Optional[Union[float, torch.Tensor]] = None,
    ):
        self.min_dist = min_dist
        self.spread = spread

        self.a: Union[float, torch.Tensor]
        self.b: Union[float, torch.Tensor]

        if a is None or b is None:
            self.a, self.b = _find_ab_params(self.spread, self.min_dist)
        if a is not None:
            self.a = a
        if b is not None:
            self.b = b

    def transform(
        self,
        reldata: relationdata.RelationData,
    ) -> relationdata.RelationData:

        if isinstance(
            reldata,
            (
                relationdata.SquareRelationTensor,
                relationdata.TriangularRelationTensor,
                relationdata.FlatRelationTensor,
            ),
        ):
            reldata.data = reldata.data.pow(2.0 * self.b)
            reldata.data = torch.pow(1.0 + self.a * reldata.data, -1.0)
        elif isinstance(
            reldata,
            (
                relationdata.SquareRelationArray,
                relationdata.TriangularRelationArray,
                relationdata.FlatRelationArray,
            ),
        ):
            reldata.data = np.power(1.0 + reldata.data ** (2.0 * self.b), -1.0)
        elif isinstance(reldata, relationdata.SparseRelationArray):
            return self.transform(reldata.to_square_array())
        elif isinstance(reldata, relationdata.NeighborRelationTuple):
            neighbors, rels = reldata.data
            rels = np.power(1.0 + rels ** (2.0 * self.b), -1.0)
            reldata.data = (neighbors, rels)
        else:
            raise TypeError(
                "Expected :class:`paradime.relationdata.RelationData`."
            )
        return reldata


@functools.cache
def _find_ab_params(spread: float, min_dist: float) -> tuple[float, float]:
    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, _ = scipy.optimize.curve_fit(curve, xv, yv)
    return params[0], params[1]


class Functional(RelationTransform):
    """Applies a function to the relation data.

    By default, this transform applies a given function to the :attr:`data`
    attribute of the :class:`paradime.relationdata.RelationData` instance in
    place and returns the transformed instance. This assumes that the transform
    does not change the data in a way that is incompatible with the
    :class:`paradime.relationdata.RelationData` subclass. The transform can
    also be applied to the whole :class:`paradime.relationdata.RelationData`
    instance by setting ``in_place`` to False. In this case, the output is that
    of the given function.

    Args:
        f: Function to be applied to the relations.
        in_place: Toggles whether the function is applied to the :attr:`data`
            attribute of the :class:`paradime.relationdata.RelationData` object
            (default), or to the :class:`paradime.relationdata.RelationData`
            itself.
        check_valid: Toggles whether a check for the transformed relation
            data's validity is performed. If ``in_place`` is set to False, no
            checks are performed regardless of this parameter.
    """

    def __init__(
        self,
        f: Callable[..., Any],
        in_place: bool = True,
        check_valid: bool = False,
    ):
        self.f = f
        self.in_place = in_place
        self.check_valid = check_valid

    def transform(
        self,
        reldata: relationdata.RelationData,
    ) -> relationdata.RelationData:

        if self.in_place:
            reldata.data = self.f(reldata.data)
            if self.check_valid:
                try:
                    type(reldata)(reldata.data)  # type: ignore
                except:
                    raise ValueError(
                        "Transformed relation data no longer compatible "
                        f"with type {type(reldata).__name__}."
                    )
            return reldata
        else:
            return self.f(reldata)
