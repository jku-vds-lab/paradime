ParaDime Specifications
=======================

In addition to the object-oriented way of specifying ParaDime routines, ParaDime also defines a grammar for YAML specifications.
This grammar is explained in detail in `our arXiv preprint <https://arxiv.org/abs/2210.04582>`_.

A specification includes the top-level fields ``relations``, ``losses``, ``training phases``, and ``derived data``. Of these two, ``losses`` and ``training phases`` are required. The following is a simple example of such a YAML specification, which specifies a parametric version of metric MDS:

.. code-block:: yaml

    relations:
      - name: dists hd
        level: global
        type: pairwise
        options:
          metric: euclidean
      - name: dists ld
        level: batch
        type: pairwise
        options:
          metric: euclidean
    losses:
      - name: mds
        type: relation
        func: mse
        keys:
        rels:
          - dists hd
          - dists ld
    training phases:
        - loss:
          components: mds

ParaDime routines can be constructed from such specifications using the :meth:`~paradime.dr.ParametricDR.from_spec` class method:

.. code-block:: python3

    dr = paradime.dr.ParametricDR.from_spec(
        <file or dict>,
        <model>,
    )

This method accepts either a string with the name of a YAML file or a dictionary (e.g., an already parsed specification). Along with the specification you also need to pass a model. Data is added later, either with the :meth:`~paradime.dr.ParametricDR.train` call or using :meth:`~paradime.dr.ParametricDR.add_data`.

The full schema of required fields and allowed values in ParaDime specifications is defined in the :mod:`~paradime.utils.parsing` module, using the `Cerberus syntax for validation schemata <https://docs.python-cerberus.org/en/stable/schemas.html>`_.