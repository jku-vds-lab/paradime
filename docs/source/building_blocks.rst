.. _building-blocks:

Building Blocks of a ParaDime Routine
=====================================

ParaDime generalizes the concept of parametric dimensionality reduction (DR) by introdcuing a flexible way of specifying all of the necessary *building blocks* of a DR routine. This general interface is provided by the :class:`~paradime.dr.ParametricDR` class located in ParaDime's core module :mod:`paradime.dr`. In this section we will go through all the building blocks that fully define what an instance of :class:`~paradime.dr.ParametricDR` does.

Overview
--------

The main steps of a ParaDime routine are the following:

#. For a given set of :ref:`training-data`, compute :ref:`relations` between data items.
#. Transform these relations using :ref:`transforms`.
#. Set up training loops organized into :ref:`training-phases`.
#. For each training phase, sample batches from the training data.
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

In ParaDime, much like in scikit-learn, you pass the training data when you call a routine's :meth:`~paradime.dr.ParametricDR.train`. Even though we only perform this call at the very end of a routine's setup, we discuss the data here in the beginning. The reason for this is that some of the building blocks we will discuss later can access different parts of the dataset, so it's important to see already now how it will be passed. Consider that ``dr`` is a :class:`~paradime.dr.ParametricDR` routine:

.. code-block:: python3

    dr.train({
        'main': your_data,
        'labels': your_labels,
    })

Notice how the data is usually a dictionary of named data tensors (in this case some main data and labels). You can also pass only a single data tensor to the :meth:`~paradime.dr.ParametricDR.train` method. In that case, ParaDime internally creates a dictionary from your data. It stores the single tensor under the ``'main'`` key. ParaDime also always adds an ``'indices'`` entry that simply enumerates your data items (unless your dataset already contained custom ``'indices'``). We'll see later how those keys are used to access the different parts of the data during the different stages of the routine.

Sometimes it might be preferrable to not wait until the :meth:`~paradime.dr.ParametricDR.train` call with defining your training data. In these cases you can already add the data to a routine earlier using the :meth:`~paradime.dr.ParametricDR.add_data` method. You then don't necessarily have to pass an argument to the :meth:`~paradime.dr.ParametricDR.train` call. Again you can pass either a single tensor-like object or a dict of tensors-like objects.

ParaDime alos defines its own :class:`~paradime.dr.Dataset` class that wraps around PyTorch's :class:`~torch.utils.data.Dataset`, but most of the time you don't need to create the :class:`~paradime.dr.Dataset` instance yourself. The two methods for registrtion above take care of it. After adding data or running the training, your routine will have a :class:`~paradime.dr.Dataset` instance as its ``dataset`` member.

In some cases, such as for initialization purposes, you might find it necessary to define a so-called derived data. This is data that is computed from other parts of the data or from computed relations. :class:`~paradime.dr.DerivedData` is defined as follows:

.. code-block:: python3

    derived = paradime.dr.DerivedData(
        func=...,
        type_key_tuples=[("data", "main"), ("rels", "foo")]
    )

Here, ``func`` is the function that tells ParaDime how to compute the data, and the ``type_key_tuples`` is a list of tuples that specifies which arguments will later be passed to that function internally. The first element of each tuple is either ``"data"`` or ``"rels"``, and the second element is the key to access the part (e.g., the ``"labels"`` defined above).


.. _relations:

Relations
---------

ParaDime distinguishes between two types of relations: **global** and **batch-wise** relations. **Global** relations are calculated for the whole dataset once before the actual training. **Batch-wise** relations are calculated for the processed batches of items during training. For both cases, you define the relations used in a routine by providing instances of any of the :class:`~paradime.relations.Relations` subclasses defined in the :mod:`paradime.relations` module.

Think of :class:`~paradime.relations.Relations` as recipes that are defined at instantiation of a :class:`~paradime.dr.ParametricDR`, but only invoked later in the routine. You pass the relations with the ``global_relations`` and ``batch_relations`` keyword parameters:

.. code-block:: python3

    dr = paradime.dr.ParametricDR(
        ...,
        global_relations = paradime.relations.PDist(),
        batch_relations = paradime.relations.DifferentiablePDist(),
        ...,
    )

In this example, ParaDime would calculate the full pairwise distances between data items for the whole dataset before training, and it would use a differentiable implementation of pairwise distances to calculate relations between data items for each batch. See the API reference for :mod:`paradime.relations` for a full list of relations.

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

Naming the relations with the keys is necessary to access them properly later on (similar to the :ref:`dataset` attributes mentioned above). Again, ParaDime internally constructs the dictionary for you if you only pass a single relation, for which uses the default key ``'rel'``.

You can compute all the global relations for a routine by calling its :meth:`~paradime.dr.ParametricDR.compute_relations` method. This stores the relations in the routine's ``global_relation_data`` dictionary in the form of :class:`~paradime.relationdata.RelationData` objects. The :meth:`~paradime.dr.ParametricDR.compute_relations` method is also called before the training automatically in case the relations haven't been computed beforehand. By default, relations are computed for the ``'data'`` attribute of your dataset (to align with the only entry when passing a single tensor-like object as ``dataset``). If you want relations to be computed for a different attribute, you can specify that by passing the attribute name to the ``data_key`` parameter in the relations' constructor. 

If you want to experiment with custom distance metrics for the relations, the predefined distance-based relations all accept a ``metric`` parameter. A more general way to customize relations is to subclass the :class:`~paradime.relations.Relations` base class and redefine its :meth:`~paradime.relations.Relations.compute_relations` method. Many customizations can also be performed using :ref:`transforms`, as explained below.

.. _transforms:

Relation Transforms
-------------------

Relation transforms are a concept inspired by existing techniques (e.g., t-SNE and UMAP), which first calculate basic relations, such as pairwise distances between data items, and then rescale and transform them into new relations (sometimes called probabilities or affinities).

You define the transform that you want to apply to relations by passing a :class:`~paradime.transforms.RelationTransform` instance to a :class:`~paradime.relations.Relations` instance with the ``transform`` keyword parameter. For example, to normalize pairwise distances, you would use:

.. code-block:: python3

    dist_norm = paradime.relations.PDist(
        transform=paradime.transforms.Normalize()
    )

If you pass a list of transforms instead, they are applied consecutively.

The easiest way to customize transforms is to use the :class:`~paradime.transforms.Functional` class, which lets you define your own function to be applied to the relation data.

.. _training-phases:

Training Phases
---------------

The training of a ParaDime routine is organized into training *phases*. A training phase is defined by a number of specifications that tell ParaDime how to sample batches from the dataset and how to optimize the model. Most importantly, each training phases has a loss specification, which are covered in detail in the section on :ref:`losses`.

There are two ways to define training phases: during instantiation of a :class:`~paradime.dr.ParametricDR` object using the ``training_phases`` keyword parameter; or at any point later using the :meth:`~paradime.dr.ParametricDR.add_training_phase` method. In the first case, you have to supply a list of :class:`paradime.dr.TrainingPhase` objects:

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

Each routine also has a default setting for the training routines attached to it. The routine's defaults are used instead of the global defaults while adding training phases. You can set the defaults either during instantiation (by passing a :class:`paradime.dr.TrainingPhase` as the ``training_defaults`` argument), or at any time later by using the :meth:`paradime.dr.ParametricDR.set_training_defaults` method. The training defaults allow you, for instance, to define a single batch size to be used in all training phases, without having to specify it each time.

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

The names of the training phases are for logging purposes.

.. _model:

Model
-----

Each training phase consists of a training loop in which a neural network model is applied to a batch of data. By default, ParaDime tries to infer the input dimensionality of the model based on any data that is already added to a routine. If no data has been added yet, you need to give ParaDime an input dimension so that it can construct a default fully connected model (see :class:`~paradime.models.FullyConnecteedEmbeddingModel`). You can control the layers of this model using the ``in_dim``, ``out_dim``, and ``hidden_dims`` keyword parameters.

.. code-block:: python3

    dr = paradime.dr.ParametricDR(
        ...,
        in_dim=100,  # or let ParaDime infer this from the dataset
        out_dim=3,  # default out_dim is 2
        hidden_dims=[100, 50],
        ...,
    )

For more control, you can directly pass a custom model (any PyTorch :class:`~torch.nn.Module`) using the ``model`` keyword argument:

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

If you create a :class:`~paradime.dr.ParametricDR` instance with ``use_cuda=True``, the model is moved to the GPU.

We plan to add mote predefined models for certain tasks to the :mod:`paradime.models` module in the future.

.. _losses:

Losses
------

Arguably the most important setting of each training phase is the loss to be used in the phase. You specify losses in a similar way to the training data, as a dictionary of named :class:`~paradime.loss.Loss` instances. But for the losses, this is done at instantiation:

.. code-block:: python3

    dr = paradime.dr.ParametricDR(
        ...,
        losses={
            "name": ...,
            ...,
        },
    )

The training phases access these losses based on their name. The names are passed to the training phases via the ``loss_keys`` argument. You can also weight losses with multiple components by passing ``loss_weights`` (a list of numbers with equal length to the list of loss keys). The loss defines what to do once a batch of data has been sampled from the dataset. Basically, for each batch the loss's ``forward`` method is applied with the following call signature:

.. code-block:: python3

    def forward(
        model,  # the routine's model
        global_relation_data,  # the dict of computed RelationData
        batch_relations,  # the dict of batch-wise Relations
        batch,  # the sampled batch
        device,  # the device on which the model is located
    ) -> torch.Tensor:

As you see, the loss receives the model, the already computed global relation data, the recipes for computing the batch-wise relations, the sampled batch of data, and the device on which the model lives.

Usually you don't have to worry about this, because ParaDime comes with four predefined types of losses that should be sufficient for most cases:

* :class:`~paradime.loss.RelationsLoss`: This is probably the most important loss for DR, because it compares subsets of the global relation data to newly computed batch-wise relations of the data processed by the model. You have to pass the comparison function to be used as the ``loss_function`` parameter. You can specify which relations are used by setting the loss's ``global_relation_key`` and ``batch_relation_key`` parameters. The default value is ``'rel'`` in both cases, to align with the default keys that are used when you only pass a single :class:`~paradmie.relations.Relations` instance. This means that, unless you use multiple relations per training phases, you don't have to care about setting these keys at all, as the defaults should be sufficient. You can also customize which model method is used to process the data by setting the ``embedding_method`` parameter; by default, the model's ``embed`` method is used. When using custom models, it is often easiest to set ``embedding_method = 'forward'``.
* :class:`~paradime.loss.ClassificationLoss`: By default, this loss applies the model's ``classify`` method to the batch of input ``'data'`` and compares the results to the dataset's ``'labels'`` using a cross-entropy loss. As the name implies, this loss is meant for classification models or supervised learning, and by itself it disregards the relations entirely. You can customize which model method, data attributes, and loss function to use through its keyword arguments.
* :class:`~paradime.loss.ReconstructionLoss`: By deafult, this loss applies the model's ``encode`` method followed by its ``decode`` method to the batch's ``'data'``. The result is compared to the input data. This loss is meant for training autoencoder models, and it can be customized in a similar way as described above.
* :class:`~paradime.loss.PositionalLoss`: This loss, by default, compares ``'data'`` batches processed by the model's ``embed`` method to a subset of ``'pos'`` data specified in the dataset. It is especially useful for training phases that should nitialize a routine.

Finally, you can combine multiple losses using a :class:`~paradime.loss.CompoundLoss`. You instantiate a compound loss with a list of losses (and an optional list of weights), and it calls and sums up the individual losses for you.

All losses keep track of their accumulated output. ParaDime calls each loss's :class:`~paradime.loss.Loss.checkpoint` method once at the end of each epoch to store the most recent accumulated value in the loss's ``history`` list. This allows you to inspect the evolution of all losses after the training. Compound losses also have a :meth:`~paradime.loss.CompoundLoss.detailed_history` method that outputs the history of each loss component multiplied by its weight. During training, the total loss is logged if you set the ``verbose`` flag to True.

Further Information
-------------------

For all the details on the classes, methods, and parameters, see the :ref:`api`. You will also get a good idea of how to use all these buildings blocks to define a variety of ParaDime routines by checking out the :ref:`examples`.