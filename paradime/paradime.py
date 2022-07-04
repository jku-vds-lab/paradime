from datetime import datetime
import warnings
from typing import Union, Callable, Literal
# from grpc import Call
from numba.core.types.scalars import Boolean
import torch
import torch.utils.data as td
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numba import jit
from sklearn.decomposition import PCA

import paradime.relations as pdrel
import paradime.modules as pdmod
import paradime.loss as pdloss
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

    def __getitem__(self, index) -> dict:
        out = {}
        for k in self.data:
            out[k] = self.data[k][index]
        return out

class NegSampledEdgeDataset(td.Dataset):
    def __init__(self, p_ij, neg_sampling_rate=5):
        self.p_ij = p_ij.tocoo()
        self.weights = p_ij.data
        self.neg_sampling_rate = neg_sampling_rate

    def __len__(self):
        return len(self.p_ij.data)
    
    def __getitem__(self, idx):
        # make nsr+1 copies of i
        rows = torch.full(
            (self.neg_sampling_rate + 1,),
            self.p_ij.row[idx],
            dtype=torch.long
        )

        #make one positive sample and nsr negative ones
        cols = torch.randint(
            self.p_ij.shape[0],
            (self.neg_sampling_rate + 1,),
            dtype=torch.long
        )
        cols[0] = self.p_ij.col[idx]

        # make simplified p_ij (0 or 1)
        p_simpl = torch.zeros(self.neg_sampling_rate + 1, dtype=torch.float32)
        p_simpl[0] = 1

        return rows, cols, p_simpl


class ParametricDR():

    def __init__(self,
        encoder: pdmod.Encoder = None,
        decoder: pdmod.Decoder = None,
        hd_relations: pdrel.Relations = None,
        ld_relations: pdrel.Relations = None
        ) -> None:

        self.encoder = encoder
        self.decoder = decoder
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

        if self.trained and self.encoder is not None:
            return self.encoder(X)
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





