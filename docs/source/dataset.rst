.. _dataset:

Training Dataset Format
=======================

The Snips NLU library leverages machine learning algorithms and some training
data in order to produce a powerful intent recognition engine.

The better your training data is, and the more accurate your NLU engine will
be. Thus, it is worth spending a bit of time to create a dataset that
matches well your use case.

Snips NLU accepts two different dataset formats. The first one, which relies
on YAML, is the preferred option if you want to create or edit a dataset
manually.
The other dataset format uses JSON and should rather be used if you plan to
create or edit datasets programmatically.

.. _yaml_format:

===========
YAML format
===========

The YAML dataset format allows you to define intents and entities using the
`YAML <http://yaml.org/about.html>`_ syntax.

.. _yaml_entity_format:

Entity
------

Here is what an entity file looks like:

.. code-block:: yaml

    # City Entity
    ---
    type: entity # allows to differentiate between entities and intents files
    name: city # name of the entity
    values:
      - london # single entity value
      - [new york, big apple] # entity value with a synonym
      - [paris, city of lights]

You can specify entity values either using single YAML scalars (e.g. ``london``),
or using lists if you want to define some synonyms (e.g.
``[paris, city of lights]``)

Here is a more comprehensive example which contains additional attributes that
are optional:

.. code-block:: yaml

    # City Entity
    ---
    type: entity
    name: city
    automatically_extensible: false # default value is true
    use_synonyms: false # default value is true
    matching_strictness: 0.8 # default value is 1.0
    values:
      - london
      - [new york, big apple]
      - [paris, city of lights]

.. _yaml_intent_format:

Intent
------

Here is the format used to describe an intent:

.. code-block:: yaml

    # searchFlight Intent
    ---
    type: intent
    name: searchFlight # name of the intent
    utterances:
      - find me a flight from [origin:city](Paris) to [destination:city](New York)
      - I need a flight leaving [date:snips/datetime](this weekend) to [destination:city](Berlin)
      - show me flights to go to [destination:city](new york) leaving [date:snips/datetime](this evening)

We use a standard markdown-like annotation syntax to annotate slots within
utterances. The ``[origin:city](Paris)`` chunk describes a slot with its three
components:

    - ``origin``: the slot name
    - ``city``: the slot type
    - ``Paris``: the slot value

Note that different slot names can share the same slot type. This is the case
for the ``origin`` and ``destination`` slot names in the previous example, which
have the same slot type ``city``.

If you are to write more than just three utterances, you can actually specify
the slot mapping explicitly in the intent file and remove it from the
utterances. This will result in simpler annotations:

.. code-block:: yaml

    # searchFlight Intent
    ---
    type: intent
    name: searchFlight
    slots:
      - name: origin
        entity: city
      - name: destination
        entity: city
      - name: date
        entity: snips/datetime
    utterances:
      - find me a flight from [origin](Paris) to [destination](New York)
      - I need a flight leaving [date](this weekend) to [destination](Berlin)
      - show me flights to go to [destination](new york) leaving [date](this evening)

.. important::

    If one of your utterances starts with ``[``, you must put it between
    double quotes to respect the YAML syntax: ``"[origin] to [destination]"``.

.. _yaml_dataset_format:

Dataset
-------

You are free to organize the yaml documents as you want. Either having one yaml
file for each intent and each entity, or gathering some documents together
(e.g. all entities together, or all intents together) in the same yaml file.
Here is the yaml file corresponding to the previous ``city`` entity and
``searchFlight`` intent merged together:

.. code-block:: yaml

    # searchFlight Intent
    ---
    type: intent
    name: searchFlight
    slots:
      - name: origin
        entity: city
      - name: destination
        entity: city
      - name: date
        entity: snips/datetime
    utterances:
      - find me a flight from [origin](Paris) to [destination](New York)
      - I need a flight leaving [date](this weekend) to [destination](Berlin)
      - show me flights to go to [destination](new york) leaving [date](this evening)

    # City Entity
    ---
    type: entity
    name: city
    values:
      - london
      - [new york, big apple]
      - [paris, city of lights]

.. important::

    If you plan to have more than one entity or intent in a YAML file, you must
    separate them using the YAML document separator: ``---``

---------------------------------------
Implicit entity values and slot mapping
---------------------------------------

In order to make the annotation process even easier, there is a mechanism that
allows to populate entity values automatically based on the entity values that
are already provided.

This results in a much simpler dataset file:

.. code-block:: yaml

    # searchFlight Intent
    ---
    type: intent
    name: searchFlight
    slots:
      - name: origin
        entity: city
      - name: destination
        entity: city
      - name: date
        entity: snips/datetime
    utterances:
      - find me a flight from [origin] to [destination]
      - I need a flight leaving [date] to [destination]
      - show me flights to go to [destination] leaving [date]

    # City Entity
    ---
    type: entity
    name: city
    values:
      - london
      - [new york, big apple]
      - [paris, city of lights]

For this to work, you need to provide at least one value for each
*custom entity*. This can be done either through an entity file, or simply by
providing an entity value in one of the annotated utterances.
Entity values are automatically generated for *builtin entities*.

Here is a final example of a valid YAML dataset leveraging implicit entity
values as well as implicit slot mapping:

.. code-block:: yaml

    # searchFlight Intent
    ---
    type: intent
    name: searchFlight
    utterances:
      - find me a flight from [origin:city](Paris) to [destination:city]
      - I need a flight leaving [date:snips/datetime] to [destination]
      - show me flights to go to [destination] leaving [date]

Note that the city entity was not provided here, but one value (``Paris``) was
provided in the first annotated utterance. The mapping between slot name and
entity is also inferred from the first two utterances.

Once your intents and entities are created using the YAML format described
previously, you can produce a dataset using the
:ref:`Command Line Interface (CLI) <cli>`:

.. code-block:: console

    snips-nlu generate-dataset en city.yaml searchFlight.yaml > dataset.json

Or alternatively if you merged the yaml documents into a single file:

.. code-block:: console

    snips-nlu generate-dataset en dataset.yaml > dataset.json

This will generate a JSON dataset and write it in the ``dataset.json`` file.
The format of the generated file is the second allowed format that is described
in the :ref:`JSON format <json_format>` section.

.. _json_format:

===========
JSON format
===========

The JSON format is the format which is eventually used by the training API. It
was designed to be easy to parse.

We created a `sample dataset`_ that you can check to better understand the
format.

There are three attributes at the root of the JSON document:

    - ``"language"``: the language of the dataset in :ref:`ISO format <languages>`
    - ``"intents"``: a dictionary mapping between intents names and intents data
    - ``"entities"``: a dictionary mapping between entities names and entities data

Here is how the entities are represented in this format:

.. code-block:: json

    {
      "entities": {
        "snips/datetime": {},
        "city": {
          "data": [
            {
              "value": "london",
              "synonyms": []
            },
            {
              "value": "new york",
              "synonyms": [
                "big apple"
              ]
            },
            {
              "value": "paris",
              "synonyms": [
                "city of lights"
              ]
            }
          ],
          "use_synonyms": true,
          "automatically_extensible": true,
          "matching_strictness": 1.0
        }
      }
    }

Note that the ``"snips/datetime"`` entity data is empty as it is a
:ref:`builtin entity <builtin_entity_resolution>`.

The intent utterances are defined using the following format:

.. code-block:: json

    {
      "data": [
        {
          "text": "find me a flight from "
        },
        {
          "text": "Paris",
          "entity": "city",
          "slot_name": "origin"
        },
        {
          "text": " to "
        },
        {
          "text": "New York",
          "entity": "city",
          "slot_name": "destination"
        }
      ]
    }

Once you have created a JSON dataset, either directly or with YAML files, you
can use it to train an NLU engine. To do so, you can use the CLI as documented
:ref:`here <training_cli>`, or the :ref:`python API <training_the_engine>`.

.. _sample dataset: https://github.com/snipsco/snips-nlu/blob/master/snips_nlu_samples/sample_dataset.json