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

As seen in the :ref:`tutorial` section, a command allows you to generate a
dataset from a :ref:`language <languages>` and a list of text files describing
:ref:`intents <intent>` and :ref:`entities <slot>`:

.. code-block:: bash

   snips-nlu generate-dataset en intent_1.txt intent_2.txt entity_1.txt

This will print a Json string to the standard output. If you want to store the
dataset directly in a Json file, you just have to pipe the previous command like
below:

.. code-block:: bash

   snips-nlu generate-dataset en intent_1.txt intent_2.txt entity_1.txt > dataset.json


Each intent file corresponds to a single intent, and the name of the file must
start with ``intent_``. The same is true for entity files, which must start
with ``entity_``.

An intent file is a text file in which each row corresponds to an utterance.
Slots, along with their corresponding slot type (entity), can be defined using
the following syntax:

.. code-block:: console

   Find me a flight from [departure:city](Paris) to [destination:city](London)
   Find me a flight from [departure:city](Moscow) [departureDate:snips/datetime](tomorrow around 9pm)

In this example, there are three different slots -- ``departure``,
``destination`` and ``departureDate`` -- and two different entities -- ``city``
and ``snips/datetime`` (which is a :ref:`builtin entity <builtin_entity_resolution>`).
Check :ref:`this section <entity_vs_slot_name>` to have more details about the
difference between slots and entities.

An entity file is a comma separated text file in which each row corresponds to
an entity value, optionally followed with its :ref:`synonyms <synonyms>`. The syntax used
is the following:

.. code-block:: console

   bedroom
   garden,yard,backyard

Here, the entity (room) has two values which are ``"bedroom"`` and ``"garden"``.
Two synonyms, ``"yard"`` and ``"backyard"``, are defined for ``"garden"``.
If a value or a synonym contains a comma, the value must be put between
double quotes ``"``.

If the value contains double quotes, it must be doubled
to be escaped like this:  ``"A value with a "","" in it"`` which corresponds
to the actual value ``A value with a "," in it``.

.. Note::

    By default entities are generated as :ref:`automatically extensible <auto_extensible>`,
    i.e. the recognition will accept additional values than the ones listed in
    the entity file. This behavior can be changed by adding at the beginning of
    the entity file the following:

    .. code-block:: bash

       # automatically_extensible=false

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