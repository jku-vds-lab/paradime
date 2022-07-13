import torch
import uuid
from typing import Optional, Literal

import paradime.relations as pdrel
import paradime.relationdata as pdreldata
import paradime.models as pdmod
from paradime.types import LossFun, Tensor
from paradime.exceptions import UnsupportedConfigurationError

class Loss(torch.nn.Module):

    _prefix = 'loss'

    def __init__(self,
        name: Optional[str] = None):
        super().__init__()

        if name is None:
            self.name = self._prefix + str(uuid.uuid4())
        else:
            self.name = name

    def forward(self,
        model: pdmod.Model,
        hd_relations: dict[str, pdreldata.RelationData],
        ld_relations: dict[str, pdrel.Relations],
        batch: dict[str, torch.Tensor],
        ) -> torch.Tensor:

        raise NotImplementedError()


class RelationLoss(Loss):

    _prefix = 'rel_loss'

    def __init__(self,
        loss_function: LossFun,
        global_relation_key: str = 'rel',
        batch_relation_key: str = 'rel',
        name: Optional[str] = None,
        ):
        super().__init__(name)

        self.loss_function = loss_function
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
            else:
                raise UnsupportedConfigurationError(
                    "RelationLoss does not support DistsFromTo with "
                    "'standard' sampling. Consider using 'negative_edge' "
                    "sampling instead."
                )

    def forward(self,
        model: pdmod.Model,
        hd_relations: dict[str, pdreldata.RelationData],
        ld_relations: dict[str, pdrel.Relations],
        batch: dict[str, torch.Tensor],
        ) -> torch.Tensor:

        if self._use_from_to:
            loss = self.loss_function(
                batch['rel'],
                ld_relations[self.batch_relation_key].compute_relations(
                    batch['from_to_data']
                ).data
            )
        else:
            loss = self.loss_function(
                hd_relations[self.global_relation_key].sub(batch['indices']),
                ld_relations[self.batch_relation_key].compute_relations(
                    model.embed(batch['data'])
                ).data
            )
        return loss
    
class ClassificationLoss(Loss):

    _prefix = 'class_loss'

    def __init__(self,
        label_key: str = 'labels',
        loss_function: LossFun = torch.nn.CrossEntropyLoss(),
        name: Optional[str] = None,
        ):
        super().__init__(name)

        self.loss_function = loss_function
        self.label_key = label_key
    
    def forward(self,
        model: pdmod.Model,
        hd_relations: dict[str, pdreldata.RelationData],
        ld_relations: dict[str, pdrel.Relations],
        batch: dict[str, torch.Tensor],
        ) -> torch.Tensor:

        return self.loss_function(
            model.classify(batch['data']),
            batch[self.label_key]
        )

class PositionLoss(Loss):

    _prefix = 'pos_loss'

    def __init__(self,
        position_key: str = 'pos',
        loss_function: LossFun = torch.nn.MSELoss(),
        name: Optional[str] = None,
        ):
        super().__init__(name)

        self.loss_function = loss_function
        self.position_key = position_key
    
    def forward(self,
        model: pdmod.Model,
        hd_relations: dict[str, pdreldata.RelationData],
        ld_relations: dict[str, pdrel.Relations],
        batch: dict[str, torch.Tensor],
        ) -> torch.Tensor:

        return self.loss_function(
            model.embed(batch['data']),
            batch[self.position_key]
        )
    
class ReconstructionLoss(Loss):

    _prefix = 'recon_loss'

    def __init__(self,
        loss_function: LossFun = torch.nn.MSELoss(),
        name: Optional[str] = None
        ):
        super().__init__(name)

        self.loss_function = loss_function

    def forward(self,
        model: pdmod.Model,
        hd_relations: dict[str, pdreldata.RelationData],
        ld_relations: dict[str, pdrel.Relations],
        batch: dict[str, torch.Tensor],
        ) -> torch.Tensor:

        return self.loss_function(
            batch['data'],
            model.decode(model.encode(batch['data'])),
        )

class CompoundLoss(Loss):

    _prefix = 'comp_loss'

    def __init__(self,
        losses: list[Loss],
        weights: Tensor,
        name: Optional[str] = None):
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
        hd_relations: dict[str, pdreldata.RelationData],
        ld_relations: dict[str, pdrel.Relations],
        batch: dict[str, torch.Tensor],
        ) -> torch.Tensor:

        total_loss = torch.tensor(0.)

        for l,w in zip(self.losses, self.weights):
            total_loss += w * l(model, hd_relations, ld_relations, batch)

        return total_loss