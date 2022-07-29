from typing import Optional, Union
import torch
import sklearn.decomposition

import paradime as prdm
from paradime import loss as pdloss
from paradime import relations as pdrel
from paradime import transforms as pdtf
from paradime import models as pdmod
from paradime.exceptions import NoDatasetRegisteredError
from paradime.types import Data

class ParametricTSNE(prdm.ParametricDR):

    def __init__(self,
        in_dim: int,
        out_dim: int = 2,
        hidden_dims: list[int] = [100, 50],
        perplexity: float = 30.,
        model: Optional[pdmod.Model] = None,
        alpha: float = 1.,
        initialization: Optional[str] = 'pca',
        epochs: int = 30,
        init_epochs: int = 5,
        batch_size: int = 100,
        data_key: str ='data',
        dataset: Optional[Union[Data, prdm.Dataset]] = None,
        use_cuda: bool = False,
        verbose: bool = False,
    ):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dims = hidden_dims
        if model is None:
            self.model = pdmod.FullyConnectedEmbeddingModel(
                self.in_dim,
                self.out_dim,
                self.hidden_dims
            )
        else:
            self.model = model
        self.perplexity = perplexity
        self.alpha = alpha
        # TODO: change type hint of alpha to Union[float, torch.Tensor]
        # once in-place bug has been fixed
        self.initialization = initialization
        self.epochs = epochs
        self.init_epochs = init_epochs
        self.batch_size = batch_size
        self.data_key = data_key

        global_rel = pdrel.NeighborBasedPDist(
            transform=pdtf.PerplexityBasedRescale(
                perplexity=perplexity
            )
        )
        batch_rel = pdrel.DifferentiablePDist(
            transform=[
                pdtf.StudentTTransform(alpha=self.alpha),
                pdtf.Normalize(),
                pdtf.Symmetrize(),
                pdtf.ToSquareTensor()
            ]
        )
        super().__init__(
            self.model,
            dataset,
            global_relations=global_rel,
            batch_relations=batch_rel,
            use_cuda=use_cuda,
            verbose=verbose,
        )

    def _prepare_training(self) -> None:
        if not self._dataset_registered:
            raise NoDatasetRegisteredError()
        self.dataset: prdm.Dataset
        if self.initialization == 'pca':
            if not hasattr(self.dataset.data, 'pca'):
                pca = torch.tensor(
                    sklearn.decomposition.PCA(
                        n_components=self.out_dim
                    ).fit_transform(
                        self.dataset.data[self.data_key]
                    ),
                    dtype=torch.float
                )
                self.dataset.data['pca'] = pca
            self.add_training_phase(
                name="pca_init",
                loss=pdloss.PositionLoss(
                    position_key='pca'
                ),
                batch_size=self.batch_size,
                n_epochs=self.init_epochs,
            )
        self.add_training_phase(
            name="embedding",
            loss=pdloss.RelationLoss(
                loss_function=pdloss.kullback_leibler_div
            ),
            batch_size=self.batch_size,
            n_epochs=self.epochs,
        )
