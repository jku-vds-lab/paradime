Installation
============

paraDime is available via PyPi through:

.. code-block:: text

    pip install paradime

paraDime requires `Numpy <https://numpy.org/>`_, `SciPy <https://scipy.org/>`_, `scikit-learn <https://scikit-learn.org/>`_, and `PyNNDescent <https://github.com/lmcinnes/pynndescent>`_  (see |req text|_ file), all of which are installed auomatically when installing paraDime.

paraDime also requires `PyTorch <https://pytorch.org/>`_, which must be installed separately. If you want to train paraDime routines on the GPU, make sure to install CUDA along with the correct ``cudatoolkit`` version. See the `PyTorch docs <https://pytorch.org/get-started/locally/>`_ for detailed installation info.

If you want to use paraDime's plotting utilities, `Matplotlib <https://matplotlib.org/>`_ has to be installed additionally.

.. |req text| replace:: ``requirements.txt``
.. _req text: https://github.com/einbandi/paradime/blob/master/requirements.txt