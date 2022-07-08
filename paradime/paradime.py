from datetime import datetime
import warnings
from typing import Union, Callable, Literal, Optional
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
from paradime.types import Tensor
from paradime.exceptions import NotTrainedError

class Dataset(td.Dataset):

    def __init__(self, data: dict[str, torch.Tensor]):

        self.data = data

        self._check_input()

    def _check_input(self) -> None:

        if not 'data' in self.data:
            raise AttributeError(
                "Dataset expects a dict with a 'data' entry."
            )

        lengths = []
        for k in self.data:
            lengths.append(len(self.data[k]))
        if len(set(lengths)) != 1:
            raise ValueError(
                "Dataset expects a dict with tensors of equal length."
            )

    def __len__(self):
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
        self.neg_sampling_rate = neg_sampling_rate

    def __len__(self):
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

        edge_data = {
            'row': rows,
            'col': cols,
            'rel': p_simpl,
        }

        remaining_data = self.dataset[idx]

        return {**remaining_data, **edge_data}



class TrainingPhase():

    def __init__(self,
        n_epochs: int = 5,
        batch_size: int = 50,
        sampling: Literal['standard', 'negative_edge'] = 'standard',
        batch_relations: Optional[dict[str, pdrel.Relations]] = None,
        loss: pdloss.Loss = pdloss.Loss(),
        optimizer: type = torch.optim.Adam,
        learning_rate: float = 0.01,
        **kwargs
    ) -> None:
        
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.sampling = sampling
        if self.sampling not in ['standard', 'negative_edge']:
            raise ValueError(
                f"Unknown sampling option {self.sampling}."
            )

        self.batch_relations = batch_relations
        self.loss = loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.kwargs = kwargs

        try:
            self.optimizer([torch.tensor([0.])])
        except:
            raise ValueError(
                f"{self.optimizer} is not a valid PyTorch optimizer."
            )

    def _prepare_loader(self,
        dataset: td.Dataset,
        relations: Optional[pdreldata.RelationData] = None,
    ) -> None:

        if self.sampling == 'negative_edge':
            if relations is None:
                raise ValueError(
                    "Negative edge-sampling requires relation data."
                )
        


    # def run(self, dataset: td.Dataset) -> None:

        


class ParametricDR():

    def __init__(self,
        model: pdmod.Model,
        hd_relations: pdrel.Relations = None,
        ld_relations: pdrel.Relations = None
        ) -> None:

        self.model = model
        self.hd_relations = hd_relations
        self.ld_relations = ld_relations



        self.trained = False

    def __call__(self,
        X: Tensor
        ) -> torch.Tensor: 
        
        return self.encode(X)

    def encode(self,
        X: Tensor
        ) -> torch.Tensor:

        X = pdutils._convert_input_to_torch(X)

        if not hasattr(self.model, 'encode'):
            raise AttributeError(
                "Model has no 'encode' method."
            )

        if self.trained:
            return self.model.encode(X)
        else:
            raise NotTrainedError(
            "DR instance is not trained yet. Call 'train' with "
            "appropriate arguments before using encoder."
            )
    

    def _prepare_dataset_and_loader(self,
        dataset: Union[td.Dataset, torch.Tensor],
        sampling: Literal['default', 'negative_edge'] = 'default',
        relations: pdrel.Relations = None,
        batch_size: int = 50
        ) -> tuple[td.Dataset, td.DataLoader]:

        if sampling == 'default':
            if isinstance(dataset, torch.Tensor):
                dataset = Dataset({
                    'data': dataset,
                    'index': torch.arange(len(dataset))
                })
            sampler = td.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle = True
            )
        elif sampling == 'negative_edge':
            # TODO: implement negative edge sampling
            if isinstance(dataset, torch.Tensor):
                dataset = Dataset({
                    'data': dataset,
                    'index': torch.arange(len(dataset))
                })
            sampler = td.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle = True
            )
        #     dataset = ...

        return dataset, sampler





