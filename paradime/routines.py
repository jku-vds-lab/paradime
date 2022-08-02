from typing import Optional, Union
import torch
import numpy as np
import sklearn.decomposition

import paradime as prdm
from paradime import loss as pdloss
from paradime import relations as pdrel
from paradime import transforms as pdtf
from paradime import models as pdmod
from paradime.exceptions import NoDatasetRegisteredError
from paradime.types import Data

class ParametricTSNE(prdm.ParametricDR):
    """A parametric version of t-SNE.

    This class provides a high-level interface for a
    :class:`paradime.paradime.ParametricDR` routine with the following
    specifications:
    - The global relations are :class:`paradime.relations.NeighborBasedPDist`,
      transformed with a :class:`paradime.transforms.PerplexityBasedRescale`
      followed by :class:`paradime.tranforms.Symmetrize`.
    - The batch relations are :class:`paradime.relations.DifferentiablePDist`,
      transformed with a :class:`paradime.relations.StudentTTransform`
      followed by :class:`paradime.transform.Normalize`.
    
    Args:
        perplexity: The desired perplexity, which can be understood as
            a smooth measure of nearest neighbors used to determine
            high-dimensional relations between data points.
        alpha: Degrees of freedom of the Student's t-disitribution used to
            calculate low-dimensional relations between data points.
        model: The model used to embed the high dimensional data.
        in_dim: The numer of dimensions of the input data, used to construct a
            default model in case none is specified. If a dataset is specified
            at instantiation, the correct value for this parameter will be
            inferred from the data dimensions.
        out_dim: The number of output dimensions (i.e., the dimensionality of
            the embedding).
        hidden_dims: Dimensions of hidden layers for the default fully
            connected model that is created if no model is specified.
        initialization: How to pretrain the model to mimic initialization of
            low-dimensional positions. By default (`'pca'`) the model is
            pretrained to output an approximation of PCA before beginning the
            main training phase.
        epochs: The number of epochs in the main training phase.
        init_epochs: The number of epochs in the pretraining (initialization).
            phase.
        batch_size: The number of items in a batch.
        data_key: The key under which the data can be found in the dataset.
        dataset: The dataset on which to perform the training. Datasets can be
            registerd after instantiation using the :meth:`register_dataset`
            class method.
        use_cuda: Whether or not to use the GPU for training.
        verbosity: Verbosity flag.
    """

    def __init__(self,
        perplexity: float = 30.,
        alpha: float = 1.,
        model: Optional[pdmod.Model] = None,
        in_dim: Optional[int] = None,
        out_dim: int = 2,
        hidden_dims: list[int] = [100, 50],        
        initialization: Optional[str] = 'pca',
        epochs: int = 30,
        init_epochs: int = 5,
        batch_size: int = 100,
        data_key: str ='data',
        dataset: Optional[Union[Data, prdm.Dataset]] = None,
        use_cuda: bool = False,
        verbose: bool = False,
    ):
        self.perplexity = perplexity
        self.alpha = alpha
        self.out_dim = out_dim
        # TODO: change type hint of alpha to Union[float, torch.Tensor]
        # once in-place bug has been fixed
        self.initialization = initialization
        self.epochs = epochs
        self.init_epochs = init_epochs
        self.batch_size = batch_size
        self.data_key = data_key

        global_rel = pdrel.NeighborBasedPDist(
            transform=[
                pdtf.PerplexityBasedRescale(
                    perplexity=self.perplexity
                ),
                pdtf.Symmetrize(),
            ]
        )
        batch_rel = pdrel.DifferentiablePDist(
            transform=[
                pdtf.StudentTTransform(alpha=self.alpha),
                pdtf.Normalize(),
                pdtf.ToSquareTensor(),
            ]
        )
        super().__init__(
            model=model,
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dims=hidden_dims,
            dataset=dataset,
            global_relations=global_rel,
            batch_relations=batch_rel,
            use_cuda=use_cuda,
            verbose=verbose,
        )

    def _prepare_training(self) -> None:
        if self.initialization == 'pca':
            pca = torch.tensor(
                sklearn.decomposition.PCA(
                    n_components=self.out_dim
                ).fit_transform(
                        self.dataset.data[self.data_key]
                ),
                dtype=torch.float
            )
            self.add_to_dataset({'pca': pca})
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
