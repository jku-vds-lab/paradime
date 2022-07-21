import torch
import uuid
from typing import Optional, Literal

import paradime.relations as pdrel
import paradime.relationdata as pdreldata
import paradime.models as pdmod
from paradime.types import LossFun, Tensor
from paradime.exceptions import UnsupportedConfigurationError

class Loss(torch.nn.Module):
    """Base class for losses.

    Custom losses should subclass this class.

    Attributes:
        name: The name of the loss (used by logging functions).
    """

    _prefix = 'loss'

    def __init__(self, name: Optional[str] = None):
        super().__init__()

        if name is None:
            self.name = self._prefix + str(uuid.uuid4())
        else:
            self.name = name

    def forward(self,
        model: pdmod.Model,
        global_relations: dict[str, pdreldata.RelationData],
        batch_relations: dict[str, pdrel.Relations],
        batch: dict[str, torch.Tensor],
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
        data_key: Key under which to find the data in the input batch.
        global_relation_key: Key under which to find the global relations.
        batch_relation_key: Key under which to find the batch-wise relations.
        name: Name of the loss (used by logging functions).
    """

    _prefix = 'rel_loss'

    def __init__(self,
        loss_function: LossFun,
        data_key: str = 'data',
        global_relation_key: str = 'rel',
        batch_relation_key: str = 'rel',
        name: Optional[str] = None,
    ):
        super().__init__(name)

        self.loss_function = loss_function
        self.data_key = data_key
        self.global_relation_key = global_relation_key
        self.batch_relation_key = batch_relation_key

        self._use_from_to = False

    def _check_sampling_and_relations(
        self,
        sampling: Literal['standard', 'negative_edge'],
        batch_relations: dict[str, pdrel.Relations],
    ) -> None:
        if isinstance(
            batch_relations[self.batch_relation_key],
            pdrel.DistsFromTo
        ):
            if sampling == 'negative_edge':
                self._use_from_to = True
                self.data_key = 'from_to_data'
                self.global_relation_key = 'rel'
            else:
                raise UnsupportedConfigurationError(
                    "RelationLoss does not support DistsFromTo with "
                    "'standard' sampling. Consider using 'negative_edge' "
                    "sampling instead."
                )

    def forward(self,
        model: pdmod.Model,
        global_relations: dict[str, pdreldata.RelationData],
        batch_relations: dict[str, pdrel.Relations],
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:

        if self._use_from_to:
            loss = self.loss_function(
                batch[self.global_relation_key],
                batch_relations[self.batch_relation_key].compute_relations(
                    batch[self.data_key]
                ).data
            )
        else:
            loss = self.loss_function(
                global_relations[self.global_relation_key].sub(
                    batch['indices']),
                batch_relations[self.batch_relation_key].compute_relations(
                    model.embed(batch[self.data_key])
                ).data
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
        global_relation_key: The key under which to find the global relations.
        batch_relation_key: The key under which to find the batch-wise
            relations.
        name: Name of the loss (used by logging functions).
    """

    _prefix = 'class_loss'

    def __init__(self,
        loss_function: LossFun = torch.nn.CrossEntropyLoss(),
        data_key: str = 'data',
        label_key: str = 'labels',
        name: Optional[str] = None,
    ):
        super().__init__(name)

        self.loss_function = loss_function
        self.data_key = data_key
        self.label_key = label_key
    
    def forward(self,
        model: pdmod.Model,
        global_relations: dict[str, pdreldata.RelationData],
        batch_relations: dict[str, pdrel.Relations],
        batch: dict[str, torch.Tensor],
        ) -> torch.Tensor:

        return self.loss_function(
            model.classify(batch[self.data_key]),
            batch[self.label_key]
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
        global_relation_key: The key under which to find the global relations.
        batch_relation_key: The key under which to find the batch-wise
            relations.
        name: Name of the loss (used by logging functions).
    """

    _prefix = 'pos_loss'

    def __init__(self,
        loss_function: LossFun = torch.nn.MSELoss(),
        data_key: str = 'data',
        position_key: str = 'pos',
        name: Optional[str] = None,
    ):
        super().__init__(name)

        self.loss_function = loss_function
        self.data_key = data_key
        self.position_key = position_key
    
    def forward(self,
        model: pdmod.Model,
        global_relations: dict[str, pdreldata.RelationData],
        batch_relations: dict[str, pdrel.Relations],
        batch: dict[str, torch.Tensor],
        ) -> torch.Tensor:

        return self.loss_function(
            model.embed(batch[self.data_key]),
            batch[self.position_key]
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
        name: Name of the loss (used by logging functions).
    """

    _prefix = 'recon_loss'

    def __init__(self,
        loss_function: LossFun = torch.nn.MSELoss(),
        data_key: str = 'data',
        name: Optional[str] = None,
        ):
        super().__init__(name)

        self.loss_function = loss_function
        self.data_key = data_key

    def forward(self,
        model: pdmod.Model,
        global_relations: dict[str, pdreldata.RelationData],
        batch_relations: dict[str, pdrel.Relations],
        batch: dict[str, torch.Tensor],
        ) -> torch.Tensor:

        return self.loss_function(
            batch[self.data_key],
            model.decode(model.encode(batch[self.data_key])),
        )

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

    _prefix = 'comp_loss'

    def __init__(self,
        losses: list[Loss],
        weights: Tensor,
        name: Optional[str] = None,
    ):
        super().__init__(name)

        self.losses = losses
        self.weights = weights

        if self.weights is None:
            self.weights = torch.ones(len(losses))
        elif len(self.weights) != len(self.losses):
            raise ValueError(
                "Size mismatch between losses and weights."
            )

    def _check_sampling_and_relations(
        self,
        sampling: Literal['standard', 'negative_edge'],
        batch_relations: dict[str, pdrel.Relations],
    ) -> None:
        for loss in self.losses:
            if isinstance(loss, RelationLoss):
                loss._check_sampling_and_relations(
                    sampling,
                    batch_relations
                )
        
    def forward(self,
        model: pdmod.Model,
        global_relations: dict[str, pdreldata.RelationData],
        batch_relations: dict[str, pdrel.Relations],
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:

        total_loss = torch.tensor(0.)

        for l,w in zip(self.losses, self.weights):
            total_loss += w * l(model, global_relations, batch_relations, batch)

        return total_loss