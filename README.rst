paraDime: A Framework for Parametric Dimensionality Reduction
=============================================================

|ReadTheDocs Badge| |License Badge| |PyPi Badge| |Black Badge|

paraDime is a modular framework for specifying and training parametric dimensionality reduction (DR) models. These models allow you to add new data points to existing low-dimensional representations of high-dimensional data. ParaDime DR models are constructed from simple building blocks (such as the relations between data points), so that experimentation with novel DR techniques becomes easy.

Installation
============

paraDime is available via PyPi through:

.. code-block:: text

    pip install paradime

paraDime requires `Numpy <https://numpy.org/>`_, `SciPy <https://scipy.org/>`_, `scikit-learn <https://scikit-learn.org/>`_, `PyNNDescent <https://github.com/lmcinnes/pynndescent>`_, and `PyTorch <https://pytorch.org/>`_ (see |req text|_ file).

In order to use PyTorch's CUDA functionality, it might be necessary to install PyTorch separately with the correct setting for the ``cudatoolkit`` option (assuming you have the CUDA Toolkit already installed). See the `PyTorch docs <https://pytorch.org/get-started/locally/>`_ for installation info.

.. |req text| replace:: ``requirements.txt``
.. _req text: https://github.com/einbandi/paradime/blob/master/requirements.txt

Documentation
=============

For a simple example with one of the predefined paraDime routines, see `Getting Started <https://paradime.readthedocs.io/en/latest/getting_started.html>`_ in the documentation.

More detailed information about how to set up cusom routines can be found in `Building Blocks of a paraDime Routine <https://paradime.readthedocs.io/en/latest/building_blocks.html>`_.

For additional examples of varying complexity, see `Examples<https://paradime.readthedocs.io/en/latest/examples.html>`_.

References
==========

.. [1] Van Der Maaten, L., Hinton, G. `“Visualizing data using t-SNE” <http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf>`__, Journal of Machine Learning Research (2008).

.. [2] LeCun, Y., Cortes, C., Burges, C.J.C. `“The MNIST database of handwritten digits” <http://yann.lecun.com/exdb/mnist/>`__ (1998).


.. |ReadTheDocs Badge| image:: https://readthedocs.org/projects/paradime/badge/?version=latest&style=flat-square
   :target: https://paradime.readthedocs.io/en/latest/index.html
   :alt: Documentation Status

.. |License Badge| image:: https://img.shields.io/github/license/einbandi/paradime/?style=flat-square
   :target: https://mit-license.org/
   :alt: License

.. |PyPi Badge| image:: https://img.shields.io/pypi/v/paradime?style=flat-square
   :target: https://pypi.org/project/paradime/
   :alt: PyPi Version

.. |Black Badge| image:: https://img.shields.io/badge/code style-black-black?&style=flat-square
   :target: https://github.com/psf/black
   :alt: Code Style
