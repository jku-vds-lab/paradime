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

.. _transforms:

Relation Transforms
-------------------

TODO

.. _training-phases:

Training Phases
---------------

TODO

.. _model:

Model
-----

TODO

.. _losses:

Losses
------

TODO
