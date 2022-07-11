from datetime import datetime
import warnings
from typing import Iterable, Union, Callable, Literal, Optional
from attr import has
# from grpc import Call
from numba.core.types.scalars import Boolean
import torch
import torch.utils.data as td
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numba import jit
from sklearn.decomposition import PCA

import paradime.relationdata as pdreldata
import paradime.relations as pdrel
import paradime.models as pdmod
import paradime.loss as pdloss
import paradime.utils as pdutils
import paradime.exceptions as pdexc
from paradime.types import Tensor

Data = Union[
    np.ndarray,
    torch.Tensor,
    dict[str, Union[
        np.ndarray,
        torch.Tensor
    ]]
]

class Dataset(td.Dataset):

    def __init__(self, data: Data) -> None:

        self.data: dict[str, torch.Tensor] = {}

        if isinstance(data, np.ndarray):
            self.data['data'] = torch.tensor(data)
        elif isinstance(data, torch.Tensor):
            self.data['data'] = data
        elif isinstance(data, dict):
            if 'data' not in data:
                raise AttributeError(
                    "Dataset expects a dict with a 'data' entry."
                )
            for k in data:
                if len(data[k]) != len(data['data']):
                    raise ValueError(
                        "Dataset dict must have values of equal length."
                    )
                if isinstance(data[k], np.ndarray):
                    self.data[k] = torch.tensor(data[k])
                elif isinstance(data[k], torch.Tensor):
                    self.data[k] = data[k]  # type: ignore
                else:
                    raise ValueError(
                        f"Value for key {k} is not a numpy array "
                        "or PyTorch tensor."
                    )
        else:
            raise ValueError(
                "Expected numpy array, PyTorch tensor, or dict "
                f"instead of {type(data)}."
            )

        if 'indices' not in self.data:
            self.data['indices'] = torch.arange(len(self))
        

    def __len__(self) -> int:
        return len(self.data['data'])

    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        out = {}
        for k in self.data:
            out[k] = self.data[k][index]
        return out


class NegSampledEdgeDataset(td.Dataset):

    def __init__(self,
        dataset: Dataset,
        relations: pdreldata.RelationData,
        neg_sampling_rate: int = 5
    ) -> None:

        self.dataset = dataset
        self.p_ij = relations.to_sparse_array().data.tocoo()
        self.weights = self.p_ij.data
        self.neg_sampling_rate = neg_sampling_rate

    def __len__(self) -> int:
        return len(self.p_ij.data)
    
    def __getitem__(self,
        idx: int
    ) -> dict[str, torch.Tensor]:
        # make nsr+1 copies of row index
        rows = torch.full(
            (self.neg_sampling_rate + 1,),
            self.p_ij.row[idx],
            dtype=torch.long
        )

        # pick nsr+1 random col indices (negative samples)
        cols = torch.randint(
            self.p_ij.shape[0],
            (self.neg_sampling_rate + 1,),
            dtype=torch.long
        )
        # set only one to an actual neighbor
        cols[0] = self.p_ij.col[idx]

        # make simplified p_ij (0 or 1)
        p_simpl = torch.zeros(
            self.neg_sampling_rate + 1,
            dtype=torch.float32
        )
        p_simpl[0] = 1

        indices = torch.tensor(np.unique(
            np.concatenate((rows.numpy(), cols.numpy()))
        ))

        edge_data = {
            'row': rows,
            'col': cols,
            'rel': p_simpl,
            'indices': indices
        }

        remaining_data = td.default_collate(
            [ self.dataset[i] for i in indices ]
        )

        return {**remaining_data, **edge_data}

def _collate_edge_batch(
    raw_batch: list[dict[str, torch.Tensor]]
) -> dict[str, torch.Tensor]:

    indices, unique_ids = np.unique(
        torch.concat([ i['indices'] for i in raw_batch ]),
        return_index=True
    )

    collated_batch = {
        'indices': torch.tensor(indices)
    }

    for k in raw_batch[0]:
        if k not in ['row', 'col', 'rel']:
            collated_batch[k] = torch.concat(
                [ i[k] for i in raw_batch]
            )[torch.tensor(unique_ids)]
        else:
            collated_batch[k] = torch.concat(
                [ i[k] for i in raw_batch ]
            )
    
    return collated_batch



class TrainingPhase():

    def __init__(self,
        name: Optional[str] = None,
        n_epochs: int = 5,
        batch_size: int = 50,
        sampling: Literal['standard', 'negative_edge'] = 'standard',
        edge_rel_key: str = 'rel',
        neg_sampling_rate: int = 5,
        loss: pdloss.Loss = pdloss.Loss(),
        optimizer: type = torch.optim.Adam,
        learning_rate: float = 0.01,
        report_interval: int = 5,
        **kwargs
    ) -> None:
        
        self.name = name
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.sampling = sampling
        if self.sampling not in ['standard', 'negative_edge']:
            raise ValueError(
                f"Unknown sampling option {self.sampling}."
            )
        self.edge_rel_key = edge_rel_key
        self.neg_sampling_rate = neg_sampling_rate

        self.loss = loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.report_interval = report_interval
        self.kwargs = kwargs

        try:
            self.optimizer([torch.tensor([0.])])
        except:
            raise ValueError(
                f"{self.optimizer} is not a valid PyTorch optimizer."
            )


RelOrRelDict = Union[
    pdrel.Relations,
    dict[str, pdrel.Relations]
]

class ParametricDR():

    def __init__(self,
        model: pdmod.Model,
        global_relations: Optional[RelOrRelDict] = None,
        batch_relations: Optional[RelOrRelDict] = None,
        training_defaults: TrainingPhase = TrainingPhase(),
        training_phases: Optional[list[TrainingPhase]] = None,
        use_cuda: bool = False,
        verbose: bool = False
        ) -> None:

        self.model = model

        # TODO: check relation input for validity
        if isinstance(global_relations, pdrel.Relations):
            self.global_relations = {
                'rel': global_relations
            }
        elif global_relations is not None:
            self.global_relations = global_relations
        else:
            self.global_relations = {}
        
        self.global_relation_data: dict[str, pdreldata.RelationData] = {}
        self._global_relations_computed = False
        
        if isinstance(batch_relations, pdrel.Relations):
            self.batch_relations = {
                'rel': batch_relations
            }
        elif batch_relations is not None:
            self.batch_relations = batch_relations
        else:
            self.batch_relations = {}

        self.training_defaults = training_defaults

        self.training_phases: list[TrainingPhase] = []
        if training_phases is not None:
            for tp in training_phases:
                self.add_training_phase(training_phase=tp)

        self.trained = False
        self.dataset: Optional[Dataset] = None
        self._dataset_registered = False

        self.use_cuda = use_cuda
        self.verbose = verbose

    def __call__(self,
        X: Tensor
    ) -> torch.Tensor: 
        
        return self.embed(X)

    def embed(self,
        X: Tensor
    ) -> torch.Tensor:

        X = pdutils._convert_input_to_torch(X)

        if not hasattr(self.model, 'embed'):
            raise AttributeError(
                "Model has no 'embed' method."
            )

        if self.trained:
            return self.model.embed(X)
        else:
            raise pdexc.NotTrainedError(
            "DR instance is not trained yet. Call 'train' with "
            "appropriate arguments before using encoder."
            )

    def set_training_defaults(self,
        training_phase: Optional[TrainingPhase] = None,
        n_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        sampling: Optional[Literal['standard', 'negative_edge']] = None,
        edge_rel_key: Optional[str] = None,
        neg_sampling_rate: Optional[int] = None,
        loss: Optional[pdloss.Loss] = None,
        optimizer: Optional[type] = None,
        learning_rate: Optional[float] = None,
        report_interval: Optional[int] = 5,
        **kwargs
    ) -> None:

        if training_phase is not None:
            self.training_defaults = training_phase
        if n_epochs is not None:
            self.training_defaults.n_epochs = n_epochs
        if batch_size is not None:
            self.training_defaults.batch_size = batch_size
        if sampling is not None:
            self.training_defaults.sampling = sampling
        if edge_rel_key is not None:
            self.training_defaults.edge_rel_key = edge_rel_key
        if neg_sampling_rate is not None:
            self.training_defaults.neg_sampling_rate = neg_sampling_rate
        if loss is not None:
            self.training_defaults.loss = loss
        if optimizer is not None:
            self.training_defaults.optimizer = optimizer
        if learning_rate is not None:
            self.training_defaults.learning_rate = learning_rate
        if report_interval is not None:
            self.training_defaults.report_interval = report_interval
        if kwargs:
            self.training_defaults.kwargs = {
                **self.training_defaults.kwargs,
                **kwargs
            }

    def add_training_phase(self,
        training_phase: Optional[TrainingPhase] = None,
        name: Optional[str] = None,
        n_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        sampling: Optional[Literal['standard', 'negative_edge']] = None,
        edge_rel_key: Optional[str] = None,
        neg_sampling_rate: Optional[int] = None,
        loss: Optional[pdloss.Loss] = None,
        optimizer: Optional[type] = None,
        learning_rate: Optional[float] = None,
        report_interval: Optional[int] = None,
        **kwargs
    ) -> None:
        if training_phase is None:
            training_phase = self.training_defaults
        assert isinstance(training_phase, TrainingPhase)

        if name is not None:
            training_phase.name = name
        if n_epochs is not None:
            training_phase.n_epochs = n_epochs
        if batch_size is not None:
            training_phase.batch_size = batch_size
        if sampling is not None:
            training_phase.sampling = sampling
        if edge_rel_key is not None:
            training_phase.edge_rel_key = edge_rel_key
        if neg_sampling_rate is not None:
            training_phase.neg_sampling_rate = neg_sampling_rate
        if loss is not None:
            training_phase.loss = loss
        if optimizer is not None:
            training_phase.optimizer = optimizer
        if learning_rate is not None:
            training_phase.learning_rate = learning_rate
        if report_interval is not None:
            training_phase.report_interval = report_interval
        if kwargs:
            training_phase.kwargs = {
                **training_phase.kwargs,
                **kwargs
            }
        
        self.training_phases.append(training_phase)

    def register_dataset(self,
        data: Data
    ) -> None:

        if isinstance(data, Dataset):
            self.dataset = data
        else:
            self.dataset = Dataset(data)

        self._dataset_registered = True

    def _compute_global_relations(self) -> None:

        if not self._dataset_registered:
            raise pdexc.NoDatasetRegisteredError(
                "Cannot compute global relations before registering dataset."
            )
        assert isinstance(self.dataset, Dataset)
        
        for k in self.global_relations:
            if self.verbose:
                pdutils.report(
                    f"Computing global relations {k}."
                )
            self.global_relation_data[k] = (
                self.global_relations[k].compute_relations(
                    self.dataset.data['data']
                ))
                
        self._global_relations_computed = True


    def _prepare_loader(self,
        training_phase: TrainingPhase
    ) -> td.DataLoader:

        if not self._dataset_registered:
            raise pdexc.NoDatasetRegisteredError(
                "Cannot prepare loader before registering dataset."
            )
        assert isinstance(self.dataset, Dataset)

        if not self._global_relations_computed:
            raise pdexc.RelationsNotComputedError(
                "Cannot prepare loader before computing global relations."
            )

        if training_phase.sampling == 'negative_edge':
            if training_phase.edge_rel_key not in self.global_relation_data:
                raise ValueError(
                    f"Global relations {training_phase.edge_rel_key} "
                    "not specified."
                )
            edge_dataset = NegSampledEdgeDataset(
                self.dataset,
                self.global_relation_data[training_phase.edge_rel_key],
                training_phase.neg_sampling_rate
            )
            sampler = td.WeightedRandomSampler(
                edge_dataset.weights,
                num_samples=training_phase.batch_size
            )
            dataloader = td.DataLoader(
                edge_dataset,
                batch_size=training_phase.batch_size,
                collate_fn=_collate_edge_batch,
                sampler=sampler
            )
        else:
            dataset = self.dataset
            dataloader = td.DataLoader(
                dataset,
                batch_size=training_phase.batch_size,
                shuffle=True
            )

        return dataloader

    def _prepare_optimizer(self,
        training_phase: TrainingPhase
    ) -> torch.optim.Optimizer:

        optimizer: torch.optim.Optimizer = training_phase.optimizer(
            self.model.parameters(),
            lr=training_phase.learning_rate,
            **training_phase.kwargs
        )

        return optimizer

    def run_training_phase(self,
        training_phase: TrainingPhase
    ) -> None:

        dataloader = self._prepare_loader(training_phase)
        optimizer = self._prepare_optimizer(training_phase)

        for epoch in range(training_phase.n_epochs):
            running_loss = 0.
            batch: dict[str, torch.Tensor]

            for batch in dataloader:

                if self.use_cuda:
                    for k in batch:
                        batch[k] = batch[k].cuda()

                optimizer.zero_grad()

                loss = training_phase.loss.forward(
                    self.model,
                    self.global_relation_data,
                    self.batch_relations,
                    batch
                )

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            if self.verbose and epoch % training_phase.report_interval == 0:
                #TODO: replace by loss reporting mechanism (GH issue #3)
                pdutils.report(
                    f"Loss after epoch {epoch}: {running_loss}"
                )

                
                
                







