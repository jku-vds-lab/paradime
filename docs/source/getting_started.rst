Getting Started
===============

paraDime is a flexible framework for specifiying parametric dimensionality reduction *routines*. A routine basically consists of a neural network, a dataset, and some instructions about what exactly paraDime should do yith your data to reduce its dimensionality.

paraDime has a flexible API with several predefined classes for each part of a routine, and each part can be fully customized by extending these existing classes. If you want to learn more about what exactly makes up a routine, see :ref:`building-blocks`.

But for now, the easiest way to get started with paraDime is to use on of the predefined routines. In the following short tutorial, we are going to train one of the predefined paraDime routines to reduce the dimensionality of data from the MNIST dataset of handwritten digits.

Importing paraDime and Loading the Dataset
------------------------------------------

First, we import the ``routines`` submodule of paraDime, which includes the predefined routines. We also import paraDime's ``utils`` subpackage, which implements a scatterplot function that we are later going to use. Finally, we import torchvision, which gives us convenient access to the MNIST dataset.

.. literalinclude:: /../../examples/predefined.py
   :language: python3
   :start-after: start-include-and-data
   :end-before: end-include-and-data

Note that we have already flattened the image data into vectors of length 784 and normalized the values to a range between 0 and 1. ``num_items`` is the size of the MNIST subset that we are going to use for training our routine.


Setting Up a Predefined Routine
-------------------------------

We now create an instance of a parametric version of the t-SNE algorithm:

.. literalinclude:: /../../examples/predefined.py
   :language: python3
   :start-after: start-define
   :end-before: end-define

When initializing a routine, paraDime only needs minimal information to set up the underlying neural network. In this case paraDime infers all the necessary information from the dataset that we pass. For more info on the default model construction, see :ref:`model`. We tell paraDime that the main part of the traingin should go on for 40 epochs, and we would like to use the GPU for training (use ``use_cuda = False`` or comment out this line, if you don't have CUDA installed.) Finally, the ``verbose`` flag tells paraDime to log some information about what is going on behind the scenes.

You might have noticed that we also pass a ``perplexity`` value, which is specific to the t-SNE algorithm.

Training the Routine and Visualizing the Results
------------------------------------------------

Since any other necessary bulding blocks are already predefined in this case, all that's left to do is to train the model. To do this, we simply call:

.. literalinclude:: /../../examples/predefined.py
   :language: python3
   :start-after: start-train
   :end-before: end-train

After the training is done, we can apply our trained model to the input data:

.. literalinclude:: /../../examples/predefined.py
   :language: python3
   :start-after: start-apply-to-train-set
   :end-before: end-apply-to-train-set

Now we can plot the dimensionality-reduced data that we used for training:

.. literalinclude:: /../../examples/predefined.py
   :language: python3
   :start-after: start-plot-train
   :end-before: end-plot-train

.. image:: ../../examples/images/predefined-1.png
   :width: 500px
   :align: center
   :alt: Scatter plot of the dimensionality-reduced training data.

Because paraDime models are parametric, you can easily apply the trained model to the whole MNIST dataset, even though our routine only ever saw a small subset of it:

.. literalinclude:: /../../examples/predefined.py
   :language: python3
   :start-after: start-apply-and-plot-rest
   :end-before: end-apply-and-plot-rest

.. image:: ../../examples/images/predefined-2.png
   :width: 500px
   :align: center
   :alt: Scatter plot of the whole dimensionality-reduced MNIST dataset.

If you want to configure our own paraDime routines, you will have to understand what was going on behind the scenes. The output log might have given you a rough idea about the different parts and steps involved in a routine. We cover all the details of it in :ref:`building-blocks`.
