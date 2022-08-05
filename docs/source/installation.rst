Installation
============

paraDime is available via PyPi through:

.. code-block:: text

    pip install paradime

paraDime requires `Numpy <https://numpy.org/>`_, `SciPy <https://scipy.org/>`_, `scikit-learn <https://scikit-learn.org/>`_, `PyNNDescent <https://github.com/lmcinnes/pynndescent>`_, and `PyTorch <https://pytorch.org/>`_ (see |req text|_ file).

In order to use PyTorch's CUDA functionality, it might be necessary to install PyTorch separately with the correct setting for the ``cudatoolkit`` option (assuming you have the CUDA Toolkit already installed). See the `PyTorch docs <https://pytorch.org/get-started/locally/>`_ for installation info.

.. |req text| replace:: ``requirements.txt``
.. _req text: https://github.com/einbandi/paradime/blob/master/requirements.txt