"""Main module of ParaDime.

The :mod:`paradime.dr` module implements the main functionality of ParaDime.
This includes the :class:`paradime.dr.ParametricDR` class, as well as
:class:`paradime.dr.Dataset` and :class:`paradime.dr.TrainingPhase`.
"""

import copy
from typing import Callable, Literal, Mapping, Optional, Type, TypeVar, Union

import numpy as np
import torch
import sklearn.decomposition
import sklearn.manifold

from paradime import exceptions
from paradime import models
from paradime import relationdata
from paradime import relations
from paradime import transforms
from paradime import loss as pdloss
from paradime.types import TensorLike, TypeKeyTuples
from paradime import utils


class DerivedData:
    """A derived dataset entry to be computed later.

    Derived dataset entries can be used to set up rules for extending existing
    datasets later based on functions acting on other dataset entries or
    global relations.

    Args:
        func: The function to compute the derived data.
        type_key_tuples: A list of (type, key) tuples, where the types can be
            ``'data'`` or ``'rels'``, and the keys are used to access the
            respective entries.
    """

    def __init__(
        self,
        func: Callable[..., TensorLike],
        type_key_tuples: TypeKeyTuples = [("data", "main")],
        **kwargs,
    ):

        self.func = func
        self.requires_relations = False
        self.type_key_tuples = type_key_tuples
        self.kwargs = kwargs

        types, keys = list(zip(*type_key_tuples))

        for t in types:
            if t == "data":
                pass
            elif t == "rels":
                self.requires_relations = True
            else:
                raise ValueError(
                    "Expected 'data' or 'rels' as argument type "
                    f"for derived entry. Found '{t}' instead."
                )


Data = Union[
    np.ndarray,
    torch.Tensor,
    Mapping[str, Union[np.ndarray, torch.Tensor]],
]


class Dataset(torch.utils.data.Dataset, utils._ReprMixin):
    """A dataset for dimensionality reduction.

    Constructs a PyTorch :class:torch.utils.data.Dataset from the given data
    in such a way that each item or batch of items is a dictionary with
    PyTorch tensors as values. If only a single numpy array or PyTorch tensor
    is passed, this data will be available under the ``'data'`` key of the
    dict. Alternatively, a dict of tensors and/or arrays can be passed, which
    allows additional data such as labels for supervised learning. By default,
    an entry for indices is added to the dict, if it is not yet included in the
    passed dict.

    Args:
        data: The data, passed either as a single numpy array or PyTorch
            tensor, or as a dictionary containing multiple arrays and/or
            tensors.
    """

    def __init__(self, data: Data):

        self.data: dict[str, torch.Tensor] = {}

        if isinstance(data, (np.ndarray, torch.Tensor)):
            data = {"main": data}
        elif not isinstance(data, dict):
            raise ValueError(
                "Expected numpy array, PyTorch tensor, or dict "
                f"instead of {type(data)}."
            )
        # else:
        #     if "data" not in data:
        #         raise AttributeError(
        #             "Dataset expects a dict with a 'data' entry."
        #         )

        for i, (k, val) in enumerate(data.items()):
            if not isinstance(val, (np.ndarray, torch.Tensor)):
                raise ValueError(
                    f"Value for key {k} is not a numpy array or "
                    "PyTorch tensor."
                )
            elif i == 0:
                self.num_items = len(val)
                self.data[k] = utils.convert.to_torch(val)
            elif len(val) != self.num_items:
                raise ValueError(
                    "Dataset dict must have values of equal length."
                )
            else:
                self.data[k] = utils.convert.to_torch(val)

        if "indices" not in self.data:
            self.data["indices"] = torch.arange(len(self))

    def __len__(self) -> int:
        return self.num_items

    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        out = {}
        for k in self.data:
            out[k] = self.data[k][index]
        return out


class NegSampledEdgeDataset(torch.utils.data.Dataset):
    """A dataset that supports negative edge sampling.

    Constructs a PyTorch :class:`torch.utils.data.Dataset` suitable for
    negative sampling from a regular :class:Dataset. The passed relation
    data, along with the negative samplnig rate ``r``, is used to inform the
    negative sampling process. Each \"item\" ``i`` of the resulting dataset
    is essentially a small batch of items, including the item ``i`` of the
    original dataset, one of it's actual neighbors, and ``r`` random other
    items that are considered to not be neighbors of ``i``. Remaining data
    from the original dataset is collated using PyTorch's
    :func:`torch.utils.data.default_collate` method.

    Args:
        data: The data in the form of a ParaDime :class:`paradime.dr.Dataset`.
        relations: A :class:`paradime.relationdata.RelationData` object with
            the edge data used for negative edge sampling.
        neg_sampling_rate: The negative sampling rate.
    """

    def __init__(
        self,
        dataset: Dataset,
        relations: relationdata.RelationData,
        neg_sampling_rate: int = 5,
        data_key: str = "main",
    ):

        self.dataset = dataset
        self.p_ij = relations.to_sparse_array().data.tocoo()
        self.weights = self.p_ij.data
        self.neg_sampling_rate = neg_sampling_rate
        self.data_key = data_key

        if self.data_key not in self.dataset.data:
            raise KeyError(
                f"Dataset does not have an attribute '{self.data_key}'."
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # the following assumes that the input relations are symmetric.
        # if they are not, the rows and cols should be shuffled

        # make nsr + 1 copies of row index
        rows = torch.full(
            (self.neg_sampling_rate + 1,), self.p_ij.row[idx], dtype=torch.long
        )

        # pick nsr + 1 random col indices (negative samples)
        cols = torch.randint(
            self.p_ij.shape[0], (self.neg_sampling_rate + 1,), dtype=torch.long
        )
        # set only one to an actual neighbor
        cols[0] = self.p_ij.col[idx]

        # make simplified p_ij (0 or 1)
        p_simpl = torch.zeros(self.neg_sampling_rate + 1, dtype=torch.float32)
        p_simpl[0] = 1

        indices = torch.tensor(
            np.unique(np.concatenate((rows.numpy(), cols.numpy())))
        )

        edge_data = {"row": rows, "col": cols, "rel": p_simpl}

        edge_data["from_to_data"] = torch.stack(
            (
                self.dataset.data[self.data_key][rows],
                self.dataset.data[self.data_key][cols],
            )
        )

        remaining_data = torch.utils.data.default_collate(
            [self.dataset[i] for i in indices]
        )

        return {**remaining_data, **edge_data}


def _collate_edge_batch(
    raw_batch: list[dict[str, torch.Tensor]]
) -> dict[str, torch.Tensor]:

    indices, unique_ids = np.unique(
        torch.concat([i["indices"] for i in raw_batch]), return_index=True
    )

    collated_batch = {"indices": torch.tensor(indices)}

    for k in raw_batch[0]:
        if k in ["row", "col", "rel"]:
            collated_batch[k] = torch.concat([i[k] for i in raw_batch])
        elif k == "from_to_data":
            collated_batch[k] = torch.concat([i[k] for i in raw_batch], dim=1)
        else:
            collated_batch[k] = torch.concat([i[k] for i in raw_batch])[
                torch.tensor(unique_ids)
            ]

    return collated_batch


class TrainingPhase(utils._ReprMixin):
    """A collection of parameter settings for a single phase in the
    training of a :class:`paradime.dr.ParametricDR` instance.

    Args:
        name: The name of the training phase.
        epochs: The number of epochs to run in this phase. In standard
            item-based sampling, the model sees every item once per epoch
            In the case of negative edge sampling, this is not guaranteed, and
            an epoch instead comprises ``batches_per_epoch`` batches (see
            parameter description below).
        batch_size: The number of items/edges in a batch. In standard
            item-based sampling, a batch has this many items, and the edges
            used for batch relations are constructed from the items. In the
            case of negative edge sampling, this is the number of sampled
            *positive* edges. The total number of edges is higher by a factor
            of ``r + 1``, where ``r`` is the negative sampling rate. The same holds
            for the number of items (apart from possible duplicates, which can
            result from the edge sampling and are removed).
        batches_per_epoch: The number of batches per epoch. This parameter
            only has an effect for negative edge sampling, where the number
            of batches per epoch is not determined by the dataset size and the
            batch size. If this parameter is set to -1 (default), an epoch
            will comprise a number of batches that leads to a total number of
            sampled *items* roughly equal to the number of items in the
            dataset. If this parameter is set to an integer, an epoch will
            instead comprise that many batches.
        sampling: The sampling strategy, which can be either ``'standard'``
            (simple item-based sampling; default) or ``'negative_edge'``
            (negative edge sampling).
        edge_rel_key: The key under which to find the global relations that
            should be used for negative edge sampling.
        neg_sampling_rate: The number of negative (i.e., non-neighbor) edges
            to sample for each real neighborhood edge.
        loss_keys: The keys under which to find the losses that should be
            minimized in this training phase.
        loss_weights: The weights for the losses. If none are specified, losses
            will be weighed equally.
        optimizer: The optmizer to use for loss minimization.
        learning_rate: The learning rate used in the optimization.
        report_interval: How often the loss should be reported during
            training, given in terms of epochs. E.g., with a setting of 5,
            the loss will be reported every 5 epochs.
        kwargs: Additional kwargs that are passed on to the optimizer.

    Attributes:
        loss: The loss constructed from the keys and weights specified above.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        epochs: int = 5,
        batch_size: int = 50,
        batches_per_epoch: int = -1,
        sampling: Literal["standard", "negative_edge"] = "standard",
        edge_rel_key: str = "rel",
        neg_sampling_rate: int = 5,
        loss_keys: list[str] = ["loss"],
        loss_weights: Optional[list[float]] = None,
        optimizer: type = torch.optim.Adam,
        learning_rate: float = 0.01,
        report_interval: int = 5,
        **kwargs,
    ):

        self.name = name
        self.epochs = epochs
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.sampling = sampling
        if self.sampling not in ["standard", "negative_edge"]:
            raise ValueError(f"Unknown sampling option {self.sampling}.")
        self.edge_rel_key = edge_rel_key
        self.neg_sampling_rate = neg_sampling_rate

        if not loss_keys:
            raise ValueError("Training phase requires at least one loss key")
        else:
            self.loss_keys = loss_keys

        if loss_weights is None:
            self.loss_weights = list(np.ones(len(self.loss_keys)))
        else:
            self.loss_weights = loss_weights

        self._loss: Optional[pdloss.Loss] = None

        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.report_interval = report_interval
        self.kwargs = kwargs

        if not issubclass(self.optimizer, torch.optim.Optimizer):
            raise ValueError(
                f"{self.optimizer} is not a valid PyTorch optimizer."
            )

    @property
    def loss(self) -> pdloss.Loss:
        if self._loss is None:
            raise exceptions.LossNotDeterminedError(
                """Attempted to access loss before it was determined."""
            )
        else:
            return self._loss

    @loss.setter
    def loss(self, ls: pdloss.Loss):
        self._loss = ls

    def _determine_loss(self, loss_dict: dict[str, pdloss.Loss]) -> None:
        if len(self.loss_keys) == 1:
            lk = self.loss_keys[0]
            if lk not in loss_dict:
                raise KeyError(f"Invalid loss key {lk}.")
            self.loss = copy.deepcopy(loss_dict[lk])
        else:
            losses: list[pdloss.Loss] = []
            for lk in self.loss_keys:
                if not lk in loss_dict:
                    raise KeyError(f"Invalid loss key {lk}.")
                else:
                    losses.append(copy.deepcopy(loss_dict[lk]))
            self.loss = pdloss.CompoundLoss(
                losses=losses,
                weights=self.loss_weights,
                name=self.name,
            )


RelOrRelDict = Union[relations.Relations, dict[str, relations.Relations]]
LossOrLossDict = Union[pdloss.Loss, dict[str, pdloss.Loss]]

_ParametricDR = TypeVar("_ParametricDR", bound="ParametricDR")


class ParametricDR(utils._ReprMixin):
    """A general parametric dimensionality reduction routine.

    Args:
        model: The PyTorch :class:`torch.nn.module` whose parameters are
            optimized during training.
        in_dim: The numer of dimensions of the input data, used to construct a
            default model in case none is specified. If a dataset is specified
            at instantiation, the correct value for this parameter will be
            inferred from the data dimensions.
        out_dim: The number of output dimensions (i.e., the dimensionality of
            the embedding).
        hidden_dims: Dimensions of hidden layers for the default fully
            connected model that is created if no model is specified.
        derived_data: A dictionary of :class:`paradime.dr.DerivedData`
            instances. These entries are computed before training, either
            before or after the global relations, depending on the options in
            the entries.
        global_relations: A single :class:`paradime.relations.Relations`
            instance or a dictionary with multiple
            :class:`paradime.relations.Relations` instances. Global relations
            are calculated once for the whole dataset before training.
        batch_relations: A single :class:`paradime.relations.Relations`
            instance or a dictionary with multiple
            :class:`paradime.relations.Relations` instances. Batch relations
            are calculated during training for each batch and are compared to
            an appropriate subset of the global relations by a
            :class:`paradime.loss.RelationLoss`.
        losses: A single :class:`paradime.loss.Loss` instance or a dictionary
            with multiple :class:`paradime.loss.Loss` instances. These losses
            are accessed by the training phases via the respective keys.
        training_defaults: A :class:`paradime.dr.TrainingPhase` object with
            settings that override the default values of all other training
            phases. This parameter is useful to avoid having to repeatedly
            set parameters to the same non-default value across training
            phases. Defaults can also be specified after isntantiation using
            the :meth:`set_training_deafults` class method.
        training_phases: A single :class:`paradime.dr.TrainingPhase` object or
            a list of :class:`paradime.dr.TrainingPhase` objects defining the
            training phases to be run. Training phases can also be added
            after instantiation using the :meth:`add_training_phase` class
            method.
        use_cuda: Whether or not to use the GPU for training.
        verbose: Verbosity flag. This setting overrides all verbosity settings
            of relations, transforms and/or losses used within the parametric
            dimensionality reduction.

    Attributes:
        device: The device on which the model is allocated (depends on the
            value specified for ``use_cuda``).
    """

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        in_dim: Optional[int] = None,
        out_dim: int = 2,
        hidden_dims: list[int] = [100, 50],
        derived_data: Optional[dict[str, DerivedData]] = None,
        global_relations: Optional[RelOrRelDict] = None,
        batch_relations: Optional[RelOrRelDict] = None,
        losses: Optional[LossOrLossDict] = None,
        training_defaults: TrainingPhase = TrainingPhase(),
        training_phases: Optional[list[TrainingPhase]] = None,
        use_cuda: bool = False,
        verbose: bool = False,
    ):

        self.verbose = verbose

        self._dataset: Optional[Dataset] = None
        self._dataset_registered = False

        if not derived_data:
            self.derived_data: dict[str, DerivedData] = {}
        else:
            self.derived_data = derived_data

        self.model: torch.nn.Module

        if model is None:
            if isinstance(in_dim, int):
                self.model = models.FullyConnectedEmbeddingModel(
                    in_dim,
                    out_dim,
                    hidden_dims,
                )
            else:
                raise ValueError(
                    "A value for 'in_dim' must be given if no model "
                    "is specified."
                )
        else:
            self.model = model

        if isinstance(global_relations, relations.Relations):
            self.global_relations = {"rel": global_relations}
        elif global_relations is not None:
            self.global_relations = global_relations
        else:
            self.global_relations = {}
        for k in self.global_relations:
            self.global_relations[k]._set_verbosity(self.verbose)

        self.global_relation_data: dict[str, relationdata.RelationData] = {}
        self._global_relations_computed = False

        if isinstance(batch_relations, relations.Relations):
            self.batch_relations = {"rel": batch_relations}
        elif batch_relations is not None:
            self.batch_relations = batch_relations
        else:
            self.batch_relations = {}
        for k in self.batch_relations:
            self.batch_relations[k]._set_verbosity(self.verbose)

        if isinstance(losses, pdloss.Loss):
            self.losses = {"loss": losses}
        elif losses is not None:
            self.losses = losses
        else:
            raise ValueError("No losses specified.")

        self.training_defaults = training_defaults

        self.training_phases: list[TrainingPhase] = []
        if training_phases is not None:
            for tp in training_phases:
                self.add_training_phase(training_phase=tp)

        self.use_cuda = use_cuda
        if use_cuda:
            if torch.cuda.is_available():
                self.model.cuda()
            else:
                raise exceptions.CUDANotAvailableError()

        self.trained = False

    @property
    def dataset(self) -> Dataset:
        if isinstance(self._dataset, Dataset):
            return self._dataset
        else:
            raise exceptions.NoDatasetRegisteredError(
                "Attempted to access a dataset, but none was registered."
            )

    @property
    def device(self) -> torch.device:
        device = torch.device("cpu")
        for p in self.model.parameters():
            device = p.device
            break
        return device

    def __call__(self, X: TensorLike) -> torch.Tensor:

        return self.embed(X)

    @classmethod
    def from_spec(
        cls: Type[_ParametricDR],
        file_or_spec: Union[str, dict],
        model: Optional[models.Model] = None,
    ) -> _ParametricDR:
        """Creates a ParaDime routine from a specification file or dictionary.

        Args:
            file_or_spec: The filename of a YAML or JSON file with a ParaDime
                specification, or a specification in dictionary form.
            model: The PyTorch :class:`torch.nn.module` to be used as the model.
        """

        spec = utils.parsing.validate_spec(file_or_spec)

        derived_data_spec = spec.get("derived data", {})
        derived_entries = _derived_entries_from_spec(derived_data_spec)

        relations_spec = spec.get("relations", {})
        g_rels, b_rels = _relations_from_spec(relations_spec)

        losses_spec = spec.get("losses", {})
        losses = _losses_from_spec(losses_spec)

        tp_spec = spec.get("training phases", {})
        training_phases = _training_phases_from_spec(tp_spec)

        dr = cls(
            model=model,
            derived_data=derived_entries,
            global_relations=g_rels,
            batch_relations=b_rels,
            losses=losses,
            training_phases=training_phases,
        )

        return dr

    def _call_model_method_by_name(
        self,
        method_name: str,
        X: TensorLike,
    ) -> torch.Tensor:

        X = utils.convert.to_torch(X)

        if self.use_cuda:
            X = X.cuda()

        if not hasattr(self.model, method_name):
            raise AttributeError(f"Model has no {method_name} method.")
        elif not callable(getattr(self.model, method_name)):
            raise ValueError(
                f"Attribute {method_name} of model is not a callable."
            )

        if self.trained:
            return getattr(self.model, method_name)(X)
        else:
            raise exceptions.NotTrainedError(
                "DR instance is not trained yet. Call 'train' with "
                "appropriate arguments before calling the model."
            )

    @torch.no_grad()
    def apply(
        self, X: TensorLike, method: Optional[str] = None
    ) -> torch.Tensor:
        """Applies the model to input data.

        Applies the model to an input tensor after first switching off
        PyTorch's automatic gradient tracking. This method also ensures that
        the resulting output tensor is on the CPU. The ``method`` parameter
        allows calling of any of the model's methods in this way, but by
        default, the model's ``__call__`` method will be used (which wraps
        around ``forward``.)

        Args:
            X: A numpy array or PyTorch tensor with the input data.
            method: The name of the model method to be applied.
        """

        if method is None:
            method = "__call__"

        return self._call_model_method_by_name(method, X).cpu()

    def embed(self, X: TensorLike) -> torch.Tensor:
        """Embeds data into the learned embedding space using the model's
        ``embed`` method.

        Args:
            X: A numpy array or PyTorch tensor with the data to be embedded.

        Returns:
            A PyTorch tensor with the embedding coordinates for the data.
        """
        return self._call_model_method_by_name("embed", X)

    def classify(self, X: TensorLike) -> torch.Tensor:
        """Classifies data using the model's ``classify`` method.

        Args:
            X: A numpy array or PyTorch tensor with the data to be classified.

        Returns:
            A PyTorch tensor with the predicted class labels for the data.
        """
        return self._call_model_method_by_name("classify", X)

    def set_training_defaults(
        self,
        training_phase: Optional[TrainingPhase] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        batches_per_epoch: Optional[int] = None,
        sampling: Optional[Literal["standard", "negative_edge"]] = None,
        edge_rel_key: Optional[str] = None,
        neg_sampling_rate: Optional[int] = None,
        loss_keys: Optional[list[str]] = None,
        loss_weights: Optional[list[float]] = None,
        optimizer: Optional[type] = None,
        learning_rate: Optional[float] = None,
        report_interval: Optional[int] = 5,
        **kwargs,
    ) -> None:
        """Sets a parametric dimensionality reduction routine's default
        training parameters.

        This methods accepts either a :class:`paradime.dr.TrainingPhase`
        instance or individual parameters passed with the same keyword syntax
        used by :class:`paradime.dr.TrainingPhase`. The specified default
        parameters will be used instead of the regular defaults when adding
        training phases.

        Args:
            training_phase: A :class:`paradime.dr.TrainingPhase` instance with
                the new default settings. Instead of this, individual
                parameters can also be passed. For a full list of training
                phase settings, see :class:`paradime.dr.TrainingPhase`.
        """
        if training_phase is not None:
            self.training_defaults = copy.deepcopy(training_phase)
        if epochs is not None:
            self.training_defaults.epochs = epochs
        if batch_size is not None:
            self.training_defaults.batch_size = batch_size
        if batches_per_epoch is not None:
            self.training_defaults.batch_size = batches_per_epoch
        if sampling is not None:
            self.training_defaults.sampling = sampling
        if edge_rel_key is not None:
            self.training_defaults.edge_rel_key = edge_rel_key
        if neg_sampling_rate is not None:
            self.training_defaults.neg_sampling_rate = neg_sampling_rate
        if loss_keys is not None:
            self.training_defaults.loss_keys = loss_keys
        if loss_weights is not None:
            self.training_defaults.loss_weights = loss_weights
        if optimizer is not None:
            self.training_defaults.optimizer = optimizer
        if learning_rate is not None:
            self.training_defaults.learning_rate = learning_rate
        if report_interval is not None:
            self.training_defaults.report_interval = report_interval
        if kwargs:
            self.training_defaults.kwargs = {
                **self.training_defaults.kwargs,
                **kwargs,
            }

    def add_training_phase(
        self,
        training_phase: Optional[TrainingPhase] = None,
        name: Optional[str] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        batches_per_epoch: Optional[int] = None,
        sampling: Optional[Literal["standard", "negative_edge"]] = None,
        edge_rel_key: Optional[str] = None,
        neg_sampling_rate: Optional[int] = None,
        loss_keys: Optional[list[str]] = None,
        loss_weights: Optional[list[float]] = None,
        optimizer: Optional[type] = None,
        learning_rate: Optional[float] = None,
        report_interval: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Adds a single training phase to a parametric dimensionality
        reduction routine.

        This methods accepts either a :class:`paradime.dr.TrainingPhase`
        instance or individual parameters passed with the same keyword syntax
        used by :class:`paradime.dr.TrainingPhase`.

        Args:
            training_phase: A :class:`paradime.dr.TrainingPhase` instance with
                the new default settings. Instead of this, individual
                parameters can also be passed. For a full list of training
                phase settings, see :class:`paradime.dr.TrainingPhase`.

        Raises:
            :class:`paradime.exceptions.UnsupportedConfigurationError`: This
                error is raised if the type of
                :class:`paradime.relation.Relations` is not compatible with the
                sampling option.
        """
        if training_phase is None:
            training_phase = copy.deepcopy(self.training_defaults)
        assert isinstance(training_phase, TrainingPhase)

        if name is not None:
            training_phase.name = name
        if epochs is not None:
            training_phase.epochs = epochs
        if batch_size is not None:
            training_phase.batch_size = batch_size
        if batches_per_epoch is not None:
            training_phase.batches_per_epoch = batches_per_epoch
        if sampling is not None:
            training_phase.sampling = sampling
        if edge_rel_key is not None:
            training_phase.edge_rel_key = edge_rel_key
        if neg_sampling_rate is not None:
            training_phase.neg_sampling_rate = neg_sampling_rate
        if loss_keys is not None:
            training_phase.loss_keys = loss_keys
        if loss_weights is not None:
            training_phase.loss_weights = loss_weights
        if optimizer is not None:
            training_phase.optimizer = optimizer
        if learning_rate is not None:
            training_phase.learning_rate = learning_rate
        if report_interval is not None:
            training_phase.report_interval = report_interval
        if kwargs:
            training_phase.kwargs = {**training_phase.kwargs, **kwargs}

        training_phase._determine_loss(self.losses)

        if isinstance(
            training_phase.loss, (pdloss.RelationLoss, pdloss.CompoundLoss)
        ):
            training_phase.loss._check_sampling_and_relations(
                training_phase.sampling, self.batch_relations
            )

        self.training_phases.append(training_phase)

    def _register_dataset(self, dataset: Data) -> None:
        """Registers a dataset for a parametric dimensionality reduction
        routine.

        Args:
            dataset: The data, passed either as a single numpy array or PyTorch
                tensor, or a dictionary containing multiple arrays and/or
                tensors.
        """
        if self.verbose:
            utils.logging.log("Initializing training dataset.")
        self._dataset = Dataset(dataset)

        self._dataset_registered = True

    def add_data(
        self, data: Mapping[str, Union[TensorLike, DerivedData]]
    ) -> None:
        """Adds data to a parametric dimensionality reduction routine.

        Tensor-like data will be added to registered dataset. If none is
        registered yet, a new one will be created and registered. Derived
        entries will be added to the routine.

        Args:
            data: A dict containing the data tensors or derived dataset entries
                to be added.
        """
        regular_data: dict[str, TensorLike] = {}
        derived_data: dict[str, DerivedData] = {}

        for k, val in data.items():
            if isinstance(val, (np.ndarray, torch.Tensor)):
                regular_data[k] = val
            elif isinstance(val, DerivedData):
                derived_data[k] = val
            else:
                raise ValueError(
                    f"Value for key {k} is not a numpy array or PyTorch tensor."
                )

        if not self._dataset_registered:
            self._register_dataset(regular_data)
        else:
            for k, val in regular_data.items():
                if self.verbose:
                    if k in self.dataset.data:
                        utils.logging.log(
                            f"Overwriting entry '{k}' in dataset."
                        )
                    else:
                        utils.logging.log(f"Adding entry '{k}' to dataset.")
                self.dataset.data[k] = utils.convert.to_torch(val)

        for k, val in derived_data.items():
            if self.verbose:
                if k in self.derived_data:
                    utils.logging.log(
                        f"Overwriting entry '{k}' in derived data."
                    )
                else:
                    utils.logging.log(f"Adding derived data entry '{k}'.")
            self.derived_data[k] = val

    def compute_derived_data(
        self,
        only: Optional[Literal["rel_based", "other"]] = None,
    ) -> None:
        """Computes the derived data entries in the registered dataset.

        After caling this function, the derived entries will be stored as
        regular entries in the routine's dataset.

        Args:
            only: If "rel_based", only those entries are computed that require
                global relations. If "other", all other entries are computed.
                By default (None), all relations are computed.
        """

        if not self._dataset_registered:
            raise exceptions.NoDatasetRegisteredError(
                "Cannot compute data before registering a dataset."
            )

        compute_rel_based = True
        compute_other = True
        if only == "rel_based":
            compute_other = False
        elif only == "other":
            compute_rel_based = False
        elif only is not None:
            raise ValueError(
                "Expected 'rel', 'other', or None for parameter 'only'."
            )

        for k, entry in self.derived_data.items():
            if (
                entry.requires_relations
                and compute_rel_based
                and not self._global_relations_computed
            ):
                raise exceptions.RelationsNotComputedError(
                    "Cannot compute relation-based derived entry before "
                    "computing global relations."
                )
            elif (entry.requires_relations and compute_rel_based) or (
                not entry.requires_relations and compute_other
            ):
                if self.verbose:
                    utils.logging.log(f"Computing derived data entry '{k}'.")

                options = entry.kwargs

                if "out_dim" not in options and entry.func in [_pca, _spectral]:
                    options["out_dim"] = 2

                selector: dict[
                    str,
                    Union[
                        dict[str, torch.Tensor],
                        dict[str, relationdata.RelationData],
                    ],
                ] = {
                    "data": self.dataset.data,
                    "rels": self.global_relation_data,
                }

                args = [selector[i][j] for i, j in entry.type_key_tuples]
                self.add_data({k: entry.func(*args, **entry.kwargs)})

    def compute_global_relations(self, force: bool = False) -> None:
        """Computes the global relations.

        The computed relation data are stored in the instance's
        ``global_relation_data`` attribute.

        Args:
            force: Whether or not to force a new computation, when relations
                have been previously computed for the same instance.
        """

        if not self._dataset_registered:
            raise exceptions.NoDatasetRegisteredError(
                "Cannot compute global relations before registering dataset."
            )
        assert isinstance(self.dataset, Dataset)

        if force or not self._global_relations_computed:

            for k in self.global_relations:
                if self.verbose:
                    utils.logging.log(f"Computing global relations '{k}'.")
                rel = self.global_relations[k]
                self.global_relation_data[k] = rel.compute_relations(
                    self.dataset.data[rel.data_key]
                )

        self._global_relations_computed = True

    def _prepare_loader(
        self, training_phase: TrainingPhase
    ) -> torch.utils.data.DataLoader:

        if not self._dataset_registered:
            raise exceptions.NoDatasetRegisteredError(
                "Cannot prepare loader before registering dataset."
            )
        assert isinstance(self.dataset, Dataset)

        if not self._global_relations_computed:
            raise exceptions.RelationsNotComputedError(
                "Cannot prepare loader before computing global relations."
            )

        if training_phase.sampling == "negative_edge":
            if training_phase.edge_rel_key not in self.global_relation_data:
                raise KeyError(
                    f"Global relations '{training_phase.edge_rel_key}' "
                    "not specified."
                )
            if training_phase.batches_per_epoch == -1:
                num_edges = max(
                    training_phase.batch_size,
                    int(
                        np.ceil(
                            len(self.dataset)
                            / (training_phase.neg_sampling_rate + 1)
                        )
                    ),
                )
            else:
                num_edges = (
                    training_phase.batch_size * training_phase.batches_per_epoch
                )
            edge_dataset = NegSampledEdgeDataset(
                self.dataset,
                self.global_relation_data[training_phase.edge_rel_key],
                training_phase.neg_sampling_rate,
            )
            sampler = torch.utils.data.WeightedRandomSampler(
                edge_dataset.weights, num_samples=num_edges
            )
            dataloader = torch.utils.data.DataLoader(
                edge_dataset,
                batch_size=training_phase.batch_size,
                collate_fn=_collate_edge_batch,
                sampler=sampler,
            )
        else:
            dataset = self.dataset
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=training_phase.batch_size, shuffle=True
            )

        return dataloader

    def _prepare_optimizer(
        self, training_phase: TrainingPhase
    ) -> torch.optim.Optimizer:

        optimizer: torch.optim.Optimizer = training_phase.optimizer(
            self.model.parameters(),
            lr=training_phase.learning_rate,
            **training_phase.kwargs,
        )

        return optimizer

    def _prepare_training(self) -> None:
        """Dummy method to inject code between instantiation and training.

        To be overwritten by subclasses. This allows, e.g., to add default
        training phases outside of a subclass's ``__init__`` but before calling
        the instance's :meth:`train` method.
        """
        pass

    def run_training_phase(self, training_phase: TrainingPhase) -> None:
        """Runs a single training phase.

        Args:
            training_phase: A :class:`paradime.dr.TrainingPhase` instance.
        """
        dataloader = self._prepare_loader(training_phase)
        optimizer = self._prepare_optimizer(training_phase)

        device = self.device

        if self.verbose:
            utils.logging.log(
                f"Beginning training phase '{training_phase.name}'."
            )

        for epoch in range(training_phase.epochs):

            batch: dict[str, torch.Tensor]
            for batch in dataloader:

                optimizer.zero_grad()

                loss = training_phase.loss(
                    self.model,
                    self.global_relation_data,
                    self.batch_relations,
                    batch,
                    device,
                )

                loss.backward()
                optimizer.step()

            training_phase.loss.checkpoint()

            if self.verbose and epoch % training_phase.report_interval == 0:
                # TODO: replace by loss reporting mechanism (GH issue #3)
                utils.logging.log(
                    f"Loss after epoch {epoch}: "
                    f"{training_phase.loss.history[-1]}"
                )

            self.trained = True

    def train(self, data: Optional[Data] = None) -> None:
        """Runs all training phases of a parametric dimensionality reduction
        routine.

        data: The training data, passed either as a single numpy array or
            PyTorch tensor, or as a dictionary containing multiple arrays
            and/or tensors.
        """

        if data is None:
            if not self._dataset_registered:
                raise exceptions.NoDatasetRegisteredError(
                    "Attempted to start training, but no data was pre-added "
                    "or passed to 'train'."
                )
        elif isinstance(data, (np.ndarray, torch.Tensor)):
            self.add_data({"main": data})
        else:
            self.add_data(data)

        self._prepare_training()
        self.compute_derived_data(only="other")
        self.compute_global_relations()
        self.compute_derived_data(only="rel_based")

        self.model.train()

        for tp in self.training_phases:
            self.run_training_phase(tp)

        self.model.eval()


def _pca(x: TensorLike, **kwargs):
    return torch.tensor(
        sklearn.decomposition.PCA(n_components=kwargs["out_dim"]).fit_transform(
            x
        ),
        dtype=torch.float,
    )


def _spectral(reldata: relationdata.RelationData, **kwargs):
    spectral = torch.tensor(
        sklearn.manifold.SpectralEmbedding(
            n_components=kwargs["out_dim"],
            affinity="precomputed",
        ).fit_transform(reldata.to_square_array().data),
        dtype=torch.float,
    )
    spectral = (spectral - spectral.mean(dim=0)) / spectral.std(dim=0)
    return spectral


def _derived_entries_from_spec(
    spec: list[dict],
) -> dict[str, DerivedData]:

    df_dict: dict[str, Callable] = {
        "pca": _pca,
        "spectral": _spectral,
    }

    derived_entries = {}
    for entry in spec:
        derived_entries[entry["name"]] = DerivedData(
            df_dict[entry["data func"]],
            entry["keys"],
            **entry.get("options", {}),
        )
    return derived_entries


def _transforms_from_spec(
    spec: list[dict],
) -> list[transforms.RelationTransform]:

    tf_dict: dict[str, type[transforms.RelationTransform]] = {
        "symmetrize": transforms.Symmetrize,
        "normalize": transforms.Normalize,
        "normalize rows": transforms.NormalizeRows,
        "perplexity": transforms.PerplexityBasedRescale,
        "t-dist": transforms.StudentTTransform,
        "connect": transforms.ConnectivityBasedRescale,
    }

    tfs: list[transforms.RelationTransform] = []

    for tfspec in spec:
        tfs.append(tf_dict[tfspec["type"]](**tfspec.get("options", {})))

    return tfs


def _relations_from_spec(
    spec: list[dict],
) -> tuple[dict[str, relations.Relations], dict[str, relations.Relations]]:

    rel_dict: dict[str, type[relations.Relations]] = {
        "precomp": relations.Precomputed,
        "pdist": relations.PDist,
        "neighbor": relations.NeighborBasedPDist,
        "pdistdiff": relations.DifferentiablePDist,
        "fromto": relations.DistsFromTo,
    }

    global_relations: dict[str, relations.Relations] = {}
    batch_relations: dict[str, relations.Relations] = {}

    for entry in spec:
        data_key = entry.get("field", "main")
        options = entry.get("options", {})
        rel_type = rel_dict[entry["type"]]
        if rel_type is not relations.Precomputed:
            options["data_key"] = data_key
        tfs = _transforms_from_spec(entry["transforms"])
        rel = rel_type(
            transform=tfs,
            **options,
        )
        if entry["level"] == "global":
            global_relations[entry["name"]] = rel
        else:
            batch_relations[entry["name"]] = rel

    return global_relations, batch_relations


def _losses_from_spec(spec: list[dict]) -> dict[str, pdloss.Loss]:

    loss_dict: dict[str, type[pdloss.Loss]] = {
        "relation": pdloss.RelationLoss,
        "classification": pdloss.ClassificationLoss,
        "reconstruction": pdloss.ReconstructionLoss,
        "position": pdloss.PositionLoss,
    }

    loss_func_dict: dict[str, Callable] = {
        "mse": torch.nn.MSELoss(),
        "kl div": pdloss.kullback_leibler_div,
        "cross entropy": torch.nn.CrossEntropyLoss(),
        "umap cross entropy": pdloss.cross_entropy_loss,
    }

    losses: dict[str, pdloss.Loss] = {}

    for entry in spec:
        if loss_dict[entry["type"]] == pdloss.RelationLoss:
            methods = entry["keys"].get("methods", ["embed"])
            losses[entry["name"]] = pdloss.RelationLoss(
                loss_function=loss_func_dict[entry["func"]],
                global_relation_key=entry["keys"]["rels"][0],
                batch_relation_key=entry["keys"]["rels"][1],
                embedding_method=methods[0],
            )
        elif loss_dict[entry["type"]] == pdloss.ClassificationLoss:
            methods = entry["keys"].get("methods", ["classify"])
            losses[entry["name"]] = pdloss.ClassificationLoss(
                loss_function=loss_func_dict[entry["func"]],
                data_key=entry["keys"]["data"][0],
                label_key=entry["keys"]["data"][1],
                classification_method=methods[0],
            )
        elif loss_dict[entry["type"]] == pdloss.ReconstructionLoss:
            methods = entry["keys"].get("methods", ["encode", "decode"])
            losses[entry["name"]] = pdloss.ReconstructionLoss(
                loss_function=loss_func_dict[entry["func"]],
                data_key=entry["keys"]["data"][0],
                encoding_method=methods[0],
                decoding_method=methods[1],
            )
        else:
            methods = entry["keys"].get("methods", ["embed"])
            losses[entry["name"]] = pdloss.PositionLoss(
                loss_function=loss_func_dict[entry["func"]],
                data_key=entry["keys"]["data"][0],
                position_key=entry["keys"]["data"][1],
                embedding_method=methods[0],
            )

    return losses


def _training_phases_from_spec(spec: list[dict]) -> list[TrainingPhase]:

    training_phases: list[TrainingPhase] = []

    for entry in spec:
        if "optimizer" in entry:
            optimizer = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}[
                entry["optimizer"]["type"]
            ]
            optim_options = entry["optimizer"].get("options", {})
        else:
            optimizer = torch.optim.Adam
            optim_options = {}
        tp = TrainingPhase(
            epochs=entry.get("epochs", 5),
            sampling=(
                "negative_edge"
                if entry["sampling"]["type"] == "edge"
                else "standard"
            ),
            loss_keys=entry["loss"]["components"],
            loss_weights=entry["loss"].get("weights"),
            optimizer=optimizer,
            **entry["sampling"].get("options", {}),
            **optim_options,
        )
        training_phases.append(tp)

    return training_phases
