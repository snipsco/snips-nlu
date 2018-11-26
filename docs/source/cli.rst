.. _cli:

Command Line Interface
======================

The easiest way to test the abilities of the Snips NLU library is through the
command line interface (CLI). The CLI is installed with the python package and
is typically used by running ``snips-nlu <command> [args]`` or alternatively
``python -m snips_nlu <command> [args]``.


.. _dataset_cli:

Creating a dataset
------------------

As seen in the :ref:`tutorial <tutorial>` section, a command allows you to generate a
dataset from a :ref:`language <languages>` and a list of YAML files containing
data for :ref:`intents <intent>` and :ref:`entities <slot>`:

.. code-block:: bash

   snips-nlu generate-dataset en my_first_intent.yaml my_second_intent.yaml my_entity.yaml

.. note::

    You don't have to use separated files for each intent and entity. You could
    for instance merge all intents together in a single ``intents.yaml`` file,
    or even merge all intents and entities in a single ``dataset.yaml`` file.

This will print a JSON string to the standard output. If you want to store the
dataset directly in a JSON file, you just have to pipe the previous command like
below:

.. code-block:: bash

   snips-nlu generate-dataset en my_first_intent.yaml my_second_intent.yaml my_entity.yaml > dataset.json

Check the :ref:`Training Dataset Format <dataset>` section for more details
about the format used to describe the training data.

.. _training_cli:

Training
--------

Once you have built a proper dataset, you can use the CLI to train an NLU
engine:

.. code-block:: bash

   snips-nlu train path/to/dataset.json path/to/persisted_engine

The first parameter corresponds to the path of the dataset file. The second
parameter is the directory where the engine should be saved after training.
The CLI takes care of creating this directory.
You can enable logs by adding a ``-v`` flag.

.. _parsing_cli:

Parsing
-------

Finally, you can use the parsing command line to test interactively the parsing
abilities of a trained NLU engine:

.. code-block:: bash

   snips-nlu parse path/to/persisted_engine

This will run a prompt allowing you to parse queries interactively.
You can also pass a single query using an optional parameter:

.. code-block:: bash

   snips-nlu parse path/to/persisted_engine -q "my query"

.. _version_cli:

Versions
--------

Two simple commands allow to print the version of the library and the version
of the NLU model:

.. code-block:: bash

   snips-nlu version
   snips-nlu model-version