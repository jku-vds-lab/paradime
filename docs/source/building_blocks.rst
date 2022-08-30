.. _building-blocks:

Building Blocks of a paraDime Routine
=====================================

paraDime generalizes the concept of parametric dimensionality reduction (DR) by introdcuing a flexible way of specifying all of the necessary *building blocks* a DR routine. This general interface is provided by the :class:`~paradime.dr.ParametricDR` class located in paraDime's core module :mod:`paradime.dr`. In this section we will go through all the building blocks that fully define what an instance of :class:`~paradime.dr.ParametricDR` does.

Overview
--------

The main steps of a paraDime routine are the following:

#. For a given :ref:`dataset`, compute :ref:`relations` between data items.
#. Transform these relations using :ref:`transforms`.
#. Set up training loops organized into :ref:`training-phases`.
#. For each training phase, sample batches from the dataset.
#. Pass the batches through a machine learning :ref:`model`.
#. Compute batch-wise relations between the processed data items.
#. Transform the batch-wise relations.
#. Compare subsets of the original relations to the batch-wise ones using :ref:`losses`.
#. Compute any other losses.
#. Use backpropagation to calculate the gradient of the losses and update the model's weights.

In the following sections, each building block is explained in more detail, along with examples on how to pass the appropriate definitions to a :class:`~paradime.dr.ParametricDR` instance.

.. _dataset:

Dataset
-------

In paraDime, you attach a dataset to a routine (also called 'registering' the dataset) before you start the training process. This is unlike most of scikit-learn's estimators, where you pass the data only when you call an estimator's ``fit`` or ``fit_transform``. The reason for this registration is that before training, paradDime needs to calculate relations between all the data items in a dataset, and it makes sense to be able to do this independently of the training step.

The easiest way to register a dataset to a paraDime routine is during instantiation:

.. code-block:: python3

    dr = paradime.dr.ParametricDR(
        ...,
        dataset=your_data,
        ...,
    )

In simple cases, ``your_data`` is just a PyTorch tensor or Numpy array. If your dataset contains any other attributes, such as labels for supervised learning, pass the data as a dictionary:

.. code-block:: python3

    dr = paradime.dr.ParametricDR(
        ...,
        dataset={
            'data': your_data,
            'labels': your_labels,
        },
        ...,
    )

In fact, paraDime will always create a dictionary from your data, even when you only pass a single tensor. It will store this tensor under the ``'data'`` key. It will also add an ``'indices'`` entry that simply enumerates your data items (unless your dataset already contained custom ``'indices'``). You will see later how those keys are used to access the different parts of your data during the different stages of the routine.

Registering the dataset during instantiation has the advantage of enabling paraDime to construct a default model by inferring the dataset's dimension. More on this in the :ref:`model` section.

You can also register the dataset at a later point using the :meth:`~paradime.dr.ParametricDR.register_dataset` method:

.. code-block:: python3

    dr.register_dataset(your_data)

Again you can pass either a single tensor-like object or a dict of tensors-like objects.

paraDime alos defines its own :class:`~paradime.dr.Dataset` class that wraps around PyTorch's :class:`~torch.utils.data.Dataset`, but most of the time you will not need to create the :class:`~paradime.dr.Dataset` instance yourself. The two methods for registrtion above will take care of it.

If at any point you want to extend an already registered dataset, use the :meth:`~paradime.dr.ParametricDR.add_to_dataset` method. Here you must always pass a dictionary of tensor-like objects:

.. code-block:: python3

    dr.add_to_dataset({'labels': your_labels})

paraDime uses this method internally in the predefined routines to add additional attributes such as PCA data to an already registered dataset.

.. _relations:

Relations
---------

paraDime distinguishes between two types of relations: **global** and **batch-wise** relations. **Global** relations are calculated for the whole dataset once before the actual training. **Batch-wise** relations are calculated for the processed batches of items during training. For both cases, you define the relations used in a routine by providing instances of any of the :class:`~paradime.relations.Relations` subclasses defined in the :mod:`paradime.relations` module.

Think of :class:`~paradime.relations.Relations` as recipes that are defined at instantiation of a :class:`~paradime.dr.ParametricDR`, but only invoked later in the routine. You pass the relations with the ``global_relations`` and ``batch_relations`` keyword parameters:

.. code-block:: python3

    dr = paradime.dr.ParametricDR(
        ...,
        global_relations = paradime.relations.PDist(),
        batch_relations = paradime.relations.DifferentiablePDist(),
        ...,
    )

In this example, paraDime would calculate the full pairwise distances between data items for the whole dataset before training, and it would use a differentiable implementation of pairwise distances to calculate relations between data items for each batch. See the API reference for :mod:`paradime.relations` for a full list of relation recipes.

In the example above, we only passed one relation object for each of the two types of relations. You may want to construct multiple different relations to combine them later or use them in different training phases. To do this, simply pass a dictionary of relations:

.. code-block:: python3

    dr = paradime.dr.ParametricDR(
        ...,
        global_relations = {
            'pdist': paradime.relations.PDist(),
            'nn_based': paradime.relations.NeighborBasedPDist(
                n_neighbors=30
            ),
        },
        ...,
    )

Naming the relations with the keys is necessary to access them properly later on (similar to the :ref:`dataset` attributes explained above). Again, paraDime internally constructs the dictionary for you if you only pass a single relation, for which it will use the default key ``'rel'``.

You can compute all the global relations for a routine by calling its :meth:`~paradime.dr.ParametricDR.compute_relations` method. This will store the relations in the routine's ``global_relation_data`` dictionary in the form of :class:`~paradime.relationdata.RelationData` objects. The :meth:`~paradime.dr.ParametricDR.compute_relations` method is also called before the training automatically in case the relations haven't been computed beforehand. By default, relations are computed for the ``'data'`` attribute of your dataset (to align with the only entry when passing a single tensor-like object as ``dataset``). If you want relations to be computed for a different attribute, you can specify that by passing the attribute name to the ``data_key`` parameter in the relations' constructor. 

If you want to experiment with custom distance metrics for the relations, the predefined distance-based relations all accept a ``metric`` parameter. A more general way to customize relations is to subclass the :class:`~paradime.relations.Relations` base class and redefine its :meth:`~paradime.relations.Relations.compute_relations` method. Many customizations can also be performed using :ref:`transforms`, as explained below.

.. _transforms:

Relation Transforms
-------------------

Relation transforms are a concept inspired by existing techniques (e.g., t-SNE and UMAP) which first calculate basic relations, such as pairwise distances between data items, and then rescale and transform them into new relations (sometimes called probabilities or affinities).

You define the transform that you want to apply to relations by passing a :class:`~paradime.transforms.RelationTransform` instance to a :class:`~paradime.relations.Relations` instance with the ``transform`` keyword parameter. For example, to normalize pairwise distances, you would use:

.. code-block:: python3

    dist_norm = paradime.relations.PDist(
        transform=paradime.transforms.Normalize()
    )

If you pass a list of transforms instead, they will be applied consecutively.

The easiest way to customize transforms is to use the :class:`~paradime.transforms.Functional` class, which lets you define your own function to be applied to the relation data.

.. _training-phases:

Training Phases
---------------

The training of a paraDime routine is organized into training *phases*. A training phase is defined by a number of specifications that tell paraDime how to sample batches from the dataset and how to optimize the model. Most importantly, each training phases has a loss specification, which will be covered in detail in the section on :ref:`losses`.

There are two ways to define training phases: during instantiation of a :class:`~paradime.dr.ParametricDR` object using the ``training_phases`` keyword parameter; or at any point later using the :meth:`~paradime.dr.ParametricDR.add_training_phase` method. In the first case, you will have to supply a list of :class:`paradime.dr.TrainingPhase` objects:

.. code-block:: python3

    dr = paradime.dr.ParametricDR(
        ...,
        training_phases = [
            paradime.dr.TrainingPhase(
                name='init',
                epochs=20,
                batch_size=100,
                ...,
            ),
            paradime.dr.TrainingPhase(
                name='main',
                epochs=30,
                ...,
            )
        ],
        ...,
    )

In the latter case you can use the keyword parameters of the :meth:`~paradime.dr.ParametricDR.add_training_phase` method (but the method also accepts a :class:`paradime.dr.TrainingPhase` object):

.. code-block:: python3

    dr = paradime.dr.ParametricDR(...)
    
    dr.add_training_phase(
        name='init',
        epochs=20,
        batch_size=100,
        ...,
    )
    dr.add_training_phase(
        name='main',
        epochs=30,
        ...,
    )

This is equivalent to the specification during instantiation shown further above.

Each routine also has a default setting for the training routines attached to it. The routine's defaults will be used instead of the global defaults while adding training phases. You can set the defaults either during instantiation (by passing a :class:`paradime.dr.TrainingPhase` as the ``training_defaults`` argument), or at any time later by using the :meth:`paradime.dr.ParametricDR.set_training_defaults` method. The training defaults allow you, for instance, to define a single batch size to be used in all training phases, without having to specify it each time.

This allows setups like the following:

.. code-block:: python3

    dr = paradime.dr.ParametricDR(...)
    
    dr.set_training_defaults(
        batch_size=200,
        learning_rate=0.02,
        ...,
    )
    dr.add_training_phase(
        name='init',
        epochs=10,
        ...,
    )
    dr.add_training_phase(
        name='main',
        epochs=20,
        learning_rate=0.01
    ...,
    )

In this example, the first phase ``'init'`` would use the specified default values 200 and 0.02 for the batch size and learning rate, respectively, and the global defaults for all other parameters that are not specifically set. The second trainig phase ``'main'`` would use a learning rate of 0.01 instead.

The names of the training phases are for logging purposes only.

.. _model:

Model
-----

Each training phase consists of a training loop in which a neural network model is applied to a batch of data. By default, paraDime tries to infer the input dimensionality of the dataset, if one is registered at instantiation, and constructs a default fully connected model (see :class:`~paradime.models.FullyConnecteedEmbeddingModel`). You can control the layers of this model using the ``in_dim``, ``out_dim``, and ``hidden_dims`` keyword parameters.

.. code-block:: python3

    dr = paradime.dr.ParametricDR(
        ...,
        in_dim=100,  # or let paraDime infer this from the dataset
        out_dim=3,  # default out_dim is 2
        hidden_dims=[100, 50],
        ...,
    )

For more control, you can directly pass your custom model (any PyTorch :class:`~torch.nn.Module`) using the ``model`` keyword argument:

.. code-block:: python3

    class MyModel(torch.nn.Module):
        def __init__(in_dim, hidden_dim, out_dim):
            super().__init__()
            self.layer1 = torch.nn.Linear(in_dim, hidden_dim)
            self.layer2 = torch.nn.Linear(hidden_dim, out_dim)

        def forward(x):
            x = self.layer1(x)
            x = torch.nn.functional.relu(x)
            x = self.layer2(x)
            x = torch.nn.functional.relu(x)
            return x

    dr = paradime.dr.ParametricDR(
        ...,
        model=MyModel(100, 50, 2),
        ...,
    )

If you create a :class:`~paradime.dr.ParametricDR` instance with ``use_cuda=True``, the model will be moved to the GPU.

Some default models for different tasks are predefined in the :mod:`paradime.models` module.

.. _losses:

Losses
------

Arguably the most important setting of each training phase is the loss to be used in the phase. You specify a loss by using a training phase's ``loss`` keyword parameter. The loss defines what to do once a batch of data has been sampled from your dataset. It does this by applying a function with the following call signature in each batch:

.. code-block:: python3

    def forward(
        model,  # the routine's model
        global_relation_data,  # the dict of computed RelationData
        batch_relations,  # the dict of batch-wise Relations
        batch,  # the sampled batch
        device,  # the device on which the model is located
    ) -> torch.Tensor:

As you see, the loss receives the model, the already computed global relation data, the recipes for computing the batch-wise relations, and the sampled batch of data.

Usually you don't have to worry about this, because paraDime comes with four predefined types of losses that should be sufficient for most cases:

* :class:`~paradime.loss.RelationsLoss`: This is probably the most important loss for DR, because it compares subsets of the global relation data to newly computed batch-wise relations of the data processed by the model. You have to pass the comparison function to be used as the ``loss_function`` parameter. You can specify which relations are used by setting the loss's ``global_relation_key`` and ``batch_relation_key`` parameters. The default value is ``'rel'`` in both cases, to align with the default keys that are used when you only pass a single :class:`~paradmie.relations.Relations` instance. This means that, unless you use multiple relations per training phases, you don't have to care about setting these keys at all, as the defaults will be sufficient. You can also customize which model method is used to process the data by setting the ``embedding_method`` parameter; by default, the model's ``embed`` method is used.
* :class:`~paradime.loss.ClassificationLoss`: By default, this loss applies the model's ``classify`` method to the batch of input ``'data'``, and compares the results to the dataset's ``'labels'`` using a cross-entropy loss. As the name implies, this loss is meant for classification models or supervised learning, and by itself it disregards the relations entirely. You can customize which model method, data attributes, and loss function to use through its keyword arguments.
* :class:`~paradime.loss.ReconstructionLoss`: By deafult, this loss applies the model's ``encode`` method, followed by its ``decode`` method, to the batch's ``'data'``. The result is compared to the input data. This loss is meant for training autoencoder models, and it can be customized in a similar way as described above.
* :class:`~paradime.loss.PositionalLoss`: This loss, by default, compares ``'data'`` batches processed by the model's ``embed`` method to a subset of ``'pos'`` data specified in the dataset. It is especially useful for training phases that should serve as initialization routine. See how the predefined parametric t-SNE routine uses this loss for PCA initialization in one of the examples.

Finally, you can combine multiple losses using a :class:`~paradime.loss.CompoundLoss`. You simply define a list of losses (and an optional list of weights), and this loss will call and sum up the individual losses.

All losses keep track of their accumulated output. paraDime calls each loss's :class:`~paradime.loss.Loss.checkpoint` method once at the end of each epoch to store the most recent accumulated value in the loss's ``history`` list. This allows you to inspect the evolution of all losses after the training. Cmpound losses also have a :meth:`~paradime.loss.CompoundLoss.detailed_history` method that outputs the history of each loss component multiplied by its weight. During training, the total loss is logged if you set the ``verbose`` flag to True.

Further Information
-------------------

For all the details on the classes, methods, and parameters, see the :ref:`api`. You will also get a good idea of how to use all these buildings blocks to define a variety of paraDime routines by checking out the :ref:`examples`.