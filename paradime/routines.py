"""Predefined ParaDime routines for existing DR techniques.

The :mod:`paradime.routines` module implements parametric versions of existing
dimensionality reduction techniques using the :class:`paradime.dr.ParametricDR`
interface.
"""

from typing import Optional, Union

import torch
import sklearn.decomposition
import sklearn.manifold

from paradime import dr
from paradime import loss as pdloss
from paradime import models
from paradime import relations
from paradime import transforms


class ParametricTSNE(dr.ParametricDR):
    """A parametric version of t-SNE.

    This class provides a high-level interface for a
    :class:`paradime.paradime.ParametricDR` routine with the following
    specifications:

    * The global relations are :class:`paradime.relations.NeighborBasedPDist`,
      transformed with a :class:`paradime.transforms.PerplexityBasedRescale`
      followed by :class:`paradime.tranforms.Symmetrize`.
    * The batch relations are :class:`paradime.relations.DifferentiablePDist`,
      transformed with a :class:`paradime.relations.StudentTTransform`
      followed by :class:`paradime.transform.Normalize`.
    * The first (optional) training phase intializes the model to approximate
      PCA (see ``intialization`` below).
    * The second training phase uses the Kullback-Leibler divergence to compare
      the relations.

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
            low-dimensional positions. By default (``'pca'``) the model is
            pretrained to output an approximation of PCA before beginning the
            main training phase.
        epochs: The number of epochs in the main training phase.
        init_epochs: The number of epochs in the pretraining (initialization).
            phase.
        batch_size: The number of items in a batch during the main training
            phase.
        init_batch_size: The number of items in a batch during the pretraining
            (initialization).
        learning_rate: The learning rate during the main training phase.
        init_learning_reate: The learning rate during the pretraining
            (initialization).
        data_key: The key under which the data can be found in the dataset.
        dataset: The dataset on which to perform the training. Datasets can be
            registerd after instantiation using the :meth:`register_dataset`
            class method.
        use_cuda: Whether or not to use the GPU for training.
        verbosity: Verbosity flag.
    """

    def __init__(
        self,
        perplexity: float = 30.0,
        alpha: float = 1.0,
        model: Optional[models.Model] = None,
        in_dim: Optional[int] = None,
        out_dim: int = 2,
        hidden_dims: list[int] = [100, 50],
        initialization: Optional[str] = "pca",
        epochs: int = 30,
        init_epochs: int = 10,
        batch_size: int = 500,
        init_batch_size: Optional[int] = None,
        learning_rate: float = 0.01,
        init_learning_rate: Optional[float] = None,
        data_key: str = "main",
        use_cuda: bool = False,
        verbose: bool = False,
    ):
        self.out_dim = out_dim
        self.initialization = initialization
        self.epochs = epochs
        self.init_epochs = init_epochs
        self.batch_size = batch_size
        if init_batch_size is None:
            self.init_batch_size = self.batch_size
        else:
            self.init_batch_size = init_batch_size
        self.learning_rate = learning_rate
        if init_learning_rate is None:
            self.init_learning_rate = self.learning_rate
        else:
            self.init_learning_rate = init_learning_rate
        self.data_key = data_key

        global_rel = relations.NeighborBasedPDist(
            transform=[
                transforms.PerplexityBasedRescale(perplexity=perplexity),
                transforms.Symmetrize(),
                transforms.Normalize(),
            ]
        )
        batch_rel = relations.DifferentiablePDist(
            transform=[
                transforms.StudentTTransform(alpha=alpha),
                transforms.Normalize(),
                transforms.ToSquareTensor(),
            ]
        )

        derived_data: dict[str, dr.DerivedData] = {}
        if self.initialization == "pca":
            derived_data["pca"] = dr.DerivedData(
                dr._pca,
                type_key_tuples=[("data", self.data_key)],
                n_components=self.out_dim,
            )

        losses = {
            "init": pdloss.PositionLoss(position_key="pca"),
            "embedding": pdloss.RelationLoss(
                loss_function=pdloss.kullback_leibler_div
            ),
        }

        init_phase = dr.TrainingPhase(
            name="pca_init",
            loss_keys=["init"],
            batch_size=self.init_batch_size,
            epochs=self.init_epochs,
            learning_rate=self.init_learning_rate,
        )

        main_phase = dr.TrainingPhase(
            name="embedding",
            loss_keys=["embedding"],
            batch_size=self.batch_size,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
        )

        super().__init__(
            model=model,
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dims=hidden_dims,
            derived_data=derived_data,
            global_relations=global_rel,
            batch_relations=batch_rel,
            losses=losses,
            training_phases=[init_phase, main_phase],
            use_cuda=use_cuda,
            verbose=verbose,
        )


class ParametricUMAP(dr.ParametricDR):
    """A parametric version of UMAP.

    This class provides a high-level interface for a
    :class:`paradime.paradime.ParametricDR` routine with the following
    specifications:

    * The global relations are :class:`paradime.relations.NeighborBasedPDist`,
      transformed with a :class:`paradime.transforms.ConnectivityBasedRescale`
      followed by :class:`paradime.tranforms.Symmetrize` with product
      subtraction.
    * The batch relations are :class:`paradime.relations.DistsFromTo` (since
      negative edge sampling is used), transformed with a
      :class:`paradime.relations.ModifiedCauchyTransform`.
    * The first (optional) training phase intializes the model to approximate
      a spectral embedding based on the global relations (see ``intialization``
      below).
    * The second training phase uses corss-entropy to compare the relations.
      This phase uses negative edge sampling.

    Args:
        n_neighbors: The desired number of neighbors used for computing the
            high-dimensional pairwise relations.
        min_dist: Effective minimum distance of points in the embedding.
        spread: Effective scale of the points in the embedding.
        a: Parameter to define the modified Cauchy distribution used to compute
            low-dimensional relations.
        b: Parameter to define the modified Cauchy distribution used to compute
            low-dimensional relations.
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
            low-dimensional positions. By default (``'spectral'``) the model is
            pretrained to output an approximation of a soectral embedding based
            on the high-dimensional relations before beginning the main
            training phase.
        epochs: The number of epochs in the main training phase.
        init_epochs: The number of epochs in the pretraining (initialization).
            phase.
        batch_size: The number of items in a batch during the main training
            phase.
        init_batch_size: The number of items in a batch during the pretraining
            (initialization).
        learning_rate: The learning rate during the main training phase.
        init_learning_reate: The learning rate during the pretraining
            (initialization).
        data_key: The key under which the data can be found in the dataset.
        dataset: The dataset on which to perform the training. Datasets can be
            registerd after instantiation using the :meth:`register_dataset`
            class method.
        use_cuda: Whether or not to use the GPU for training.
        verbosity: Verbosity flag.
    """

    def __init__(
        self,
        n_neighbors: int = 30,
        min_dist: float = 0.01,
        spread: float = 1.0,
        a: Optional[float] = None,
        b: Optional[float] = None,
        model: Optional[models.Model] = None,
        in_dim: Optional[int] = None,
        out_dim: int = 2,
        hidden_dims: list[int] = [100, 50],
        initialization: Optional[str] = "spectral",
        epochs: int = 30,
        init_epochs: int = 5,
        batch_size: int = 10,
        negative_sampling_rate: int = 5,
        init_batch_size: int = 100,
        learning_rate: float = 0.005,
        init_learning_rate: float = 0.05,
        data_key: str = "main",
        dataset: Optional[Union[dr.Data, dr.Dataset]] = None,
        use_cuda: bool = False,
        verbose: bool = False,
    ):
        self.out_dim = out_dim
        self.initialization = initialization
        self.epochs = epochs
        self.init_epochs = init_epochs
        self.batch_size = batch_size
        self.negative_sampling_rate = negative_sampling_rate
        self.init_batch_size = init_batch_size
        self.learning_rate = learning_rate
        self.init_learning_rate = init_learning_rate
        self.data_key = data_key

        global_rel = relations.NeighborBasedPDist(
            transform=[
                transforms.ConnectivityBasedRescale(n_neighbors=n_neighbors),
                transforms.Symmetrize(subtract_product=True),
            ]
        )
        batch_rel = relations.DistsFromTo(
            transform=[
                transforms.ModifiedCauchyTransform(
                    min_dist=min_dist,
                    spread=spread,
                    a=a,
                    b=b,
                )
            ]
        )

        derived_data: dict[str, dr.DerivedData] = {}
        if self.initialization == "spectral":
            derived_data["spectral"] = dr.DerivedData(
                dr._spectral,
                type_key_tuples=[("rels", "rel")],
                n_components=self.out_dim,
            )

        losses = {
            "spectral_init": pdloss.PositionLoss(position_key="spectral"),
            "embedding": pdloss.RelationLoss(
                loss_function=pdloss.cross_entropy_loss
            ),
        }

        init_phase = dr.TrainingPhase(
            name="spectral_init",
            loss_keys=["spectral_init"],
            batch_size=self.init_batch_size,
            epochs=self.init_epochs,
            learning_rate=self.init_learning_rate,
        )

        main_phase = dr.TrainingPhase(
            name="embedding",
            loss_keys=["embedding"],
            sampling="negative_edge",
            neg_sampling_rate=self.negative_sampling_rate,
            batch_size=self.batch_size,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
        )

        super().__init__(
            model=model,
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dims=hidden_dims,
            derived_data=derived_data,
            global_relations=global_rel,
            batch_relations=batch_rel,
            losses=losses,
            training_phases=[init_phase, main_phase],
            use_cuda=use_cuda,
            verbose=verbose,
        )
