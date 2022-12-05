"""Losses for ParaDime routines.

The :mod:`paradime.loss` module implements the specification of losses for
ParaDime routines. The supported losses are
:class:`paradime.loss.RelationLoss`,
:class:`paradime.loss.ClassificationLoss`,
:class:`paradime.loss.ReconstructionLoss`, and
:class:`paradime.loss.CompoundLoss`.
"""

from typing import Optional, Literal, Union

import numpy as np
import torch

from paradime import exceptions
from paradime import models
from paradime import relationdata
from paradime import relations
from paradime.types import BinaryTensorFun
from paradime import utils


class Loss(torch.nn.Module):
    """Base class for losses.

    Custom losses should subclass this class.

    Attributes:
        name: The name of the loss (used by logging functions).
    """

    _prefix = "loss"

    def __init__(self, name: Optional[str] = None):
        super().__init__()

        if name is None:
            self._name = self._prefix + "_" + str(id(self))
        else:
            self._name = name

        self._accumulated: float = 0.0
        self.history: list[float] = []

        self._history_hook = self.register_forward_hook(self._add_last)

    @property
    def name(self) -> str:
        return self._name

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        # redefine to call super to improve type hinting
        return super().__call__(*args, **kwargs)

    def _add_last(
        self,
        model: torch.nn.Module,
        input: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        self._accumulated += float(output.detach().cpu().item())

    def checkpoint(self) -> None:
        """Create a checkpoint of the most recent accumulated loss.

        Appends the value of the most recent accumulated loss to the loss's
        ``history`` attribute. If the loss is a
        :class:`paradime.loss.CompoundLoss`, checkpoints are also created for
        each individual loss.
        """
        self.history.append(self._accumulated)
        self._accumulated = 0.0

    def forward(
        self,
        model: models.Model,
        global_relations: dict[str, relationdata.RelationData],
        batch_relations: dict[str, relations.Relations],
        batch: dict[str, torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """Apply the loss to a batch of input data.

        Args:
            model: The :class:`torch.nn.module` used to embed, classify, or
                reconstruct a batch of input data.
            global_relations: A dictionary with
                :class:`paradime.relationdata.RelationData` computed for the
                whole dataset.
            batch_relations: A dictionary with
                :class:`paradime.relations.Relations` to be computed for the
                batch of input data.
            batch: A batch of input data as a dictionary of PyTorch tensors.
            device: The device that all relevant tensors will be moved to.

        Returns:
            A single-item PyTorch tensor with the computed loss.
        """
        raise NotImplementedError()


class RelationLoss(Loss):
    """A loss that compares batch-wise relation data against a subset of
    global relation data.

    This loss applies a specified loss function to a subset of pre-computed
    global relations and the batch-wise relations found under specified keys,
    respectively. Batch-wise relations are computed from embedded coordinates
    by applying the model's :meth:`embed` method to the specified data entry
    of the input batch.

    Args:
        loss_function: The loss function to be applied.
        global_relation_key: Key under which to find the global relations.
        batch_relation_key: Key under which to find the batch-wise relations.
        embedding_method: The model method to be used for embedding the batch
            of input data.
        name: Name of the loss (used by logging functions).
    """

    _prefix = "rel_loss"

    def __init__(
        self,
        loss_function: BinaryTensorFun,
        global_relation_key: str = "rel",
        batch_relation_key: str = "rel",
        embedding_method: str = "embed",
        normalize_sub: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(name)

        self.loss_function = loss_function
        self.global_relation_key = global_relation_key
        self.batch_relation_key = batch_relation_key
        self.embedding_method = embedding_method
        self.normalize_sub = normalize_sub

        self._use_from_to = False

    def _check_sampling_and_relations(
        self,
        sampling: Literal["standard", "negative_edge"],
        batch_relations: dict[str, relations.Relations],
    ) -> None:
        if isinstance(
            batch_relations[self.batch_relation_key], relations.DistsFromTo
        ):
            if sampling == "negative_edge":
                self._use_from_to = True
                self.data_key = "from_to_data"
            else:
                raise exceptions.UnsupportedConfigurationError(
                    "RelationLoss does not support DistsFromTo with "
                    "'standard' sampling. Consider using 'negative_edge' "
                    "sampling instead."
                )

    def forward(
        self,
        model: models.Model,
        global_relations: dict[str, relationdata.RelationData],
        batch_relations: dict[str, relations.Relations],
        batch: dict[str, torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:

        batch_rel = batch_relations[self.batch_relation_key]
        data_key = batch_rel.data_key

        embed = getattr(model, self.embedding_method)

        if self._use_from_to:
            loss = self.loss_function(
                batch[self.global_relation_key].to(device),
                batch_rel.compute_relations(
                    embed(batch[self.data_key].to(device))
                ).data,  # type: ignore
            )
        else:
            global_rel_sub = (
                global_relations[self.global_relation_key]
                .sub(batch["indices"])
                .to(device)
            )
            if self.normalize_sub:
                global_rel_sub = global_rel_sub / global_rel_sub.sum()
            loss = self.loss_function(
                global_rel_sub,
                batch_rel.compute_relations(
                    embed(batch[data_key].to(device))
                ).data,  # type: ignore
            )
        return loss


class ClassificationLoss(Loss):
    """A loss that compares predicted class labels against ground truth labels.

    This loss compares predicted class labels to ground truth labels in a batch
    using a specified loss function (cross-entropy by default). Class labels
    are predicted by applying the model's :meth:`classify` method to the
    specified data entry of the input batch.

    Args:
        loss_function: The loss function to be applied.
        data_key: The key under which to find the data in the input batch.
        label_key: The key under which ground truth labels are stored in the
            input batch.
        classification_method: The model method to be used for classifying the
            batch of input data.
        name: Name of the loss (used by logging functions).
    """

    _prefix = "class_loss"

    def __init__(
        self,
        loss_function: BinaryTensorFun = torch.nn.CrossEntropyLoss(),
        data_key: str = "main",
        label_key: str = "labels",
        classification_method: str = "classify",
        name: Optional[str] = None,
    ):
        super().__init__(name)

        self.loss_function = loss_function
        self.data_key = data_key
        self.label_key = label_key
        self.classification_method = classification_method

    def forward(
        self,
        model: models.Model,
        global_relations: dict[str, relationdata.RelationData],
        batch_relations: dict[str, relations.Relations],
        batch: dict[str, torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:

        classify = getattr(model, self.classification_method)

        return self.loss_function(
            classify(batch[self.data_key].to(device)),
            batch[self.label_key].to(device),
        )


class PositionLoss(Loss):
    """A loss that compares embedding coordinates to given positions.

    This loss compares embedding coordiantes to given ground-truth coordinates
    in a batch using a specified loss function (mean-square-error by default).
    Embedding positions are computed by applying the model's :meth:`embed`
    method to the specified data entry of the input batch.

    Args:
        loss_function: The loss function to be applied.
        data_key: The key under which to find the data in the input batch.
        position_key: The key under which the ground truth positions are stored
            in the input batch.
        embedding_method: The model method to be used for embedding the batch
            of input data.
        name: Name of the loss (used by logging functions).
    """

    _prefix = "pos_loss"

    def __init__(
        self,
        loss_function: BinaryTensorFun = torch.nn.MSELoss(),
        data_key: str = "main",
        position_key: str = "pos",
        embedding_method: str = "embed",
        name: Optional[str] = None,
    ):
        super().__init__(name)

        self.loss_function = loss_function
        self.data_key = data_key
        self.position_key = position_key
        self.embedding_method = embedding_method

    def forward(
        self,
        model: models.Model,
        global_relations: dict[str, relationdata.RelationData],
        batch_relations: dict[str, relations.Relations],
        batch: dict[str, torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:

        embed = getattr(model, self.embedding_method)

        return self.loss_function(
            embed(batch[self.data_key].to(device)),
            batch[self.position_key].to(device),
        )


class ReconstructionLoss(Loss):
    """A simple reconstruction loss for auto-encoding data.

    This loss compares reconstructed data to input data in a batch using a
    specified loss function (mean-square-error by default). Reconstructed
    data is computed by applying the model's :meth:`decode` and :meth:`encode`
    methods subsequently to the specified data entry of the input batch.

    Args:
        loss_function: The loss function to be applied.
        data_key: The key under which to find the data in the input batch.
        encoding_method: The model method to be used for encoding the batch of
            input data.
        decoding_method: The model method to be used for decoding the encoded
            batch of input data.
        name: Name of the loss (used by logging functions).
    """

    _prefix = "recon_loss"

    def __init__(
        self,
        loss_function: BinaryTensorFun = torch.nn.MSELoss(),
        data_key: str = "main",
        encoding_method: str = "encode",
        decoding_method: str = "decode",
        name: Optional[str] = None,
    ):
        super().__init__(name)

        self.loss_function = loss_function
        self.data_key = data_key
        self.encoding_method = encoding_method
        self.decoding_method = decoding_method

    def forward(
        self,
        model: models.Model,
        global_relations: dict[str, relationdata.RelationData],
        batch_relations: dict[str, relations.Relations],
        batch: dict[str, torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:

        data = batch[self.data_key].to(device)
        encode = getattr(model, self.encoding_method)
        decode = getattr(model, self.decoding_method)

        return self.loss_function(data, decode(encode(data)))


class CompoundLoss(Loss):
    """A weighted sum of multiple losses.

    Args:
        losses: A list of :class:`paradime.loss.Loss` instances to be
            summed.
        weights: A list of weights to multiply the losses with. Must be of the
            same length as the list of losses. If no weights are specified,
            all losses are weighted equally.
        name: Name of the loss (used by logging functions).
    """

    _prefix = "comp_loss"

    def __init__(
        self,
        losses: list[Loss],
        weights: Union[np.ndarray, torch.Tensor, list[float], None] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name)

        self.losses = losses

        if weights is None:
            weights = torch.ones(len(losses))
        elif len(weights) != len(self.losses):
            raise ValueError("Size mismatch between losses and weights.")

        self.weights = utils.convert.to_torch(weights)

    def checkpoint(self) -> None:
        super().checkpoint()
        for l in self.losses:
            l.checkpoint()

    def detailed_history(self) -> torch.Tensor:
        """Returns a detailed history of the compound loss.

        Returns:
            A PyTorch tensor with the history of each loss component multiplied
            by its weight.
        """
        histories = torch.tensor([loss.history for loss in self.losses])
        return self.weights[:, None] * histories

    def _check_sampling_and_relations(
        self,
        sampling: Literal["standard", "negative_edge"],
        batch_relations: dict[str, relations.Relations],
    ) -> None:
        for loss in self.losses:
            if isinstance(loss, RelationLoss):
                loss._check_sampling_and_relations(sampling, batch_relations)

    def forward(
        self,
        model: models.Model,
        global_relations: dict[str, relationdata.RelationData],
        batch_relations: dict[str, relations.Relations],
        batch: dict[str, torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:

        total_loss = torch.tensor(0.0).to(device)

        for loss, w in zip(self.losses, self.weights.to(device)):
            loss_val = loss(
                model, global_relations, batch_relations, batch, device
            )
            total_loss += w * loss_val

        return total_loss


def kullback_leibler_div(
    p: torch.Tensor, q: torch.Tensor, epsilon: float = 1.0e-7
) -> torch.Tensor:
    """Kullback-Leibler divergence.

    To be used as a loss function in the :class:`paradime.loss.RelationLoss`
    of a parametric DR routine.

    Args:
        p: Input tensor containing (a batch of) probabilities.
        q: Input tensor containing (a batch of) probabilities.
        epsilon: Small constant used to avoid numerical errors caused by
            near-zero probability values.

    Returns:
        The Kullback-Leibler divergence of the two input tensors, divided by
        the number of items in the batch.
    """
    eps = torch.tensor(epsilon, dtype=p.dtype)
    kl_matr = torch.mul(p, torch.log(p + eps) - torch.log(q + eps))
    kl_matr.fill_diagonal_(0.0)
    return torch.sum(kl_matr) / len(p)


def cross_entropy_loss(
    p: torch.Tensor, q: torch.Tensor, epsilon: float = 1.0e-7
) -> torch.Tensor:
    """Cross-entropy loss as used by UMAP.

    To be used as a loss function in the :class:`paradime.loss.RelationLoss`
    of a parametric DR routine.

    Args:
        p: Input tensor containing (a batch of) probabilities.
        q: Input tensor containing (a batch of) probabilities.
        epsilon: Small constant used to avoid numerical errors caused by
            near-zero probability values.

    Returns:
        The cross-entropy loss of the two input tensors, divided by the number
        items in the batch.
    """
    attraction = -1.0 * p * torch.log(torch.clamp(q, min=epsilon, max=1.0))
    repulsion = (
        -1.0 * (1.0 - p) * torch.log(torch.clamp(1 - q, min=epsilon, max=1.0))
    )
    loss = attraction + repulsion
    return torch.sum(loss) / len(p)
