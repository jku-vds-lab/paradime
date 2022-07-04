from turtle import forward
from typing import Callable
import torch
import uuid

import paradime.relations as pdrel
import paradime.relationdata as pdreldata
import paradime.modules as pdmod

class Loss(torch.nn.Module):

    _prefix = 'loss'

    def __init__(self,
        name: str = None):
        super().__init__()

        if name is None:
            self.name = self._prefix + str(uuid.uuid4())
        else:
            self.name = name

    def forward(self,
        model: torch.nn.Module,
        hd_relations: pdreldata.RelationData,
        ld_relations: pdrel.Relations,
        batch: dict[str, torch.Tensor]
        ) -> torch.Tensor:

        raise NotImplementedError()


class RelationLoss(Loss):

    _prefix = 'rel_loss'

    def __init__(self,
        metric = Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        name: str = None,
        ):
        super().__init__(name)

        self.metric = metric

    def forward(self,
        model: torch.nn.Module,
        hd_relations: pdreldata.RelationData,
        ld_relations: pdrel.Relations,
        batch: dict[str, torch.Tensor]
        ) -> torch.Tensor:


        assert isinstance(batch['indices'], torch.IntTensor)

        return self.metric(
            hd_relations.sub(batch['indices']),
            ld_relations.compute_relations(model.encode(batch['data']))
        )

    