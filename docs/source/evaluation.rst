.. _evaluation:

Evaluation
==========

The ``snips-nlu`` library provides two CLI commands to compute metrics and
evaluate the quality of your NLU engine.

Cross Validation Metrics
------------------------

You can compute `cross validation`_ metrics on a given dataset by running the
following command:

.. code-block:: bash

   snips-nlu cross-val-metrics path/to/dataset.json path/to/metrics.json --include_errors

This will produce a JSON metrics report that will be stored in the ``path/to/metrics.json`` file.
This report contains:

- a `confusion matrix`_
- `F1`_, `precision and recall`_ of intent classification and slot filling for each intent, as well as globally
- parsing errors, if ``--include_errors`` was specified

You can check the CLI help for the exhaustive list of options:

.. code-block:: bash

   snips-nlu cross-val-metrics --help

Train / Test metrics
--------------------

Alternatively, you can compute metrics in a classical train / test fashion by
running the following command:

.. code-block:: bash

   snips-nlu train-test-metrics path/to/train_dataset.json path/to/test_dataset.json path/to/metrics.json

This will produce a similar metrics report to the one before.

You can check the CLI help for the exhaustive list of options:

.. code-block:: bash

   snips-nlu train-test-metrics --help


.. _cross validation: https://en.wikipedia.org/wiki/Cross-validation_(statistics)
.. _confusion matrix: https://en.wikipedia.org/wiki/Confusion_matrix
.. _precision and recall: https://en.wikipedia.org/wiki/Precision_and_recall
.. _F1: https://en.wikipedia.org/wiki/F1_score