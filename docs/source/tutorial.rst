.. _tutorial:

Tutorial
========

In this section, we will build an NLU assistant for home automation tasks. It
will be able to understand queries about lights and thermostats. More precisely
our assistant will contain three :ref:`intents <intent>`:

- ``turnLightOn``
- ``turnLightOff``
- ``setTemperature``

The first two intents will be about turning on and off the lights in a specific
room. Thus, these intents will have one :ref:`slot` which will be the ``room``.
The third intent will let you control the temperature of a specific room, thus
it will have two slots: the ``roomTemperature`` and the ``room``.

The first step is to create an appropriate dataset for this task.

.. _dataset:

Snips dataset format
--------------------

The format used by Snips to describe the input data is designed to be simple to
parse as well as easy to read.

We created a `sample dataset`_ that you can check to better understand the
format.

You have three options to create your dataset. You can build it manually by
respecting the format used in the sample, you can also use the
:ref:`dataset creation CLI <dataset_cli>` included in the lib, or alternatively
you can use `chatito`_ a DSL tool for dataset generation.

We will go for the second option here and start by creating three files
corresponding to our three intents and one entity file corresponding to the
``room`` entity:

- ``intent_turnLightOn.txt``
- ``intent_turnLightOff.txt``
- ``intent_setTemperature.txt``
- ``entity_room.txt``

The name of each file is important as the tool will map it to the intent or
entity name. In particular, the prefixes ``intent_`` and ``entity_`` are
required in order to distinguish intents from entity files.

Let's add training examples for the first intent by inserting the following
lines in the first file, ``intent_turnLightOn.txt``:

.. code-block:: console

    Turn on the lights in the [room:room](kitchen)
    give me some light in the [room:room](bathroom) please
    Can you light up the [room:room](living room) ?
    switch the [room:room](bedroom)'s lights on please

We use a standard markdown-like annotation syntax to annotate slots within
utterances. The ``[room:room]`` chunks describe the slot with its two
components: :ref:`the slot name and the entity <entity_vs_slot_name>`. In our
case we used the same value, ``room``, to describe both. The parts with
parenthesis, like ``(kitchen)``, correspond to the text value of the slot.

Let's move on to the second intent, and insert this into
``intent_turnLightOff.txt``:

.. code-block:: console

    Turn off the lights in the [room:room](entrance)
    turn the [room:room](bathroom)'s light out please
    switch off the light the [room:room](kitchen), will you?
    Switch the [room:room](bedroom)'s lights off please

And now the last file, ``intent_setTemperature.txt``:

.. code-block:: console

    Set the temperature to [roomTemperature:snips/temperature](19 degrees) in the [room:room](bedroom)
    please set the [room:room](living room)'s temperature to [roomTemperature:snips/temperature](twenty two degrees celsius)
    I want [roomTemperature:snips/temperature](75 degrees fahrenheit) in the [room:room](bathroom) please
    Can you increase the temperature to [roomTemperature:snips/temperature](22 degrees) ?

As you can see here, we used a new slot, ``[room_temperature:snips/temperature]``,
whose name is ``roomTemperature`` and whose type is ``snips/temperature``. The slot
type used here is a :ref:`builtin entity <builtin_entity_resolution>`. It
allows you to resolve the temperature values properly.

Let's move to the ``entity_room.txt`` entity file:

.. code-block:: console

    bedroom
    living room,main room
    garden,yard,backyard

The entity file is a comma (``,``) separated file. Each line corresponds to an
entity value followed by its potential :ref:`synonyms <synonyms>`.

We are now ready to generate our dataset:

.. code-block:: bash

    snips-nlu generate-dataset en intent_turnLightOn.txt intent_turnLightOff.txt intent_setTemperature.txt entity_room.txt > dataset.json

.. note::

    We used ``en`` as the language here but other languages are supported,
    please check the :ref:`languages` section to know more.

Now, the ``"entities"`` part of the generated json looks like that:

.. code-block:: json

    {
      "entities": {
        "room": {
          "automatically_extensible": true,
          "data": [
            {
              "synonyms": [],
              "value": "bedroom"
            },
            {
              "synonyms": [
                "main room"
              ],
              "value": "living room"
            },
            {
              "synonyms": [
                "yard",
                "backyard"
              ],
              "value": "garden"
            }
          ],
          "matching_strictness": 1.0,
          "use_synonyms": true
        },
        "snips/temperature": {}
      }
    }

You can see that both entities from the intent utterances and from the ``room``
entity file were added.

By default, the ``room`` entity is set to be
:ref:`automatically extensible <auto_extensible>` but in our case we don't want
to handle any entity value that would not be part of the dataset, so we set
this attribute to ``false``.
Moreover, we are going to add some rooms that were not in the previous sentences
and that we want our assistant to cover. Additionally, we add some
:ref:`synonyms <synonyms>`. Finally, the entities part looks like that:

.. code-block:: json

    {
      "entities": {
        "room": {
          "automatically_extensible": false,
          "data": [
            {
              "synonyms": [],
              "value": "bathroom"
            },
            {
              "synonyms": [
                "sleeping room"
              ],
              "value": "bedroom"
            },
            {
              "synonyms": [
                "main room",
                "lounge"
              ],
              "value": "living room"
            },
            {
              "synonyms": [
                "yard",
                "backyard"
              ],
              "value": "garden"
            }
          ],
          "matching_strictness": 1.0,
          "use_synonyms": true
        },
        "snips/temperature": {}
      }
    }


We don't need to edit the ``snips/temperature`` entity as it is a builtin
entity.

Now that we have our dataset ready, let's move to the next step which is to
create an NLU engine.

The Snips NLU Engine
--------------------

The main API of Snips NLU is an object called a :class:`.SnipsNLUEngine`. This
engine is the one you will train and use for parsing.

The simplest way to create an NLU engine is the following:

.. code-block:: python

    from snips_nlu import SnipsNLUEngine

    default_engine = SnipsNLUEngine()

In this example the engine was created with default parameters which, in
many cases, will be sufficient.

However, in some cases it may be required to tune the engine a bit and provide
a customized configuration. Typically, different languages may require
different sets of features. You can check the :class:`.NLUEngineConfig` to get
more details about what can be configured.

We have built a list of `default configurations`_, one per supported language,
that have some language specific enhancements. In this tutorial we will use the
`english one`_.

Before training the engine, note that you need to load language specific
resources used to improve performance with the :func:`.load_resources` function.

.. code-block:: python

    import io
    import json

    from snips_nlu import SnipsNLUEngine, load_resources
    from snips_nlu.default_configs import CONFIG_EN

    load_resources(u"en")

    engine = SnipsNLUEngine(config=CONFIG_EN)

At this point, we can try to parse something:

.. code-block:: python

    engine.parse(u"Please give me some lights in the entrance !")

That will raise a ``NotTrained`` error, as we did not train the engine with
the dataset that we created.


Training the engine
-------------------

In order to use the engine we created, we need to *train* it or *fit* it with
the dataset we generated earlier:

.. code-block:: python

    with io.open("dataset.json") as f:
        dataset = json.load(f)

    engine.fit(dataset)


Parsing
-------

We are now ready to parse:

.. code-block:: python

    parsing = engine.parse(u"Hey, lights on in the lounge !")
    print(json.dumps(parsing, indent=2))

You should get the following output (with a slightly different ``probability``
value):

.. code-block:: json

    {
      "input": "Hey, lights on in the lounge !",
      "intent": {
        "intentName": "turnLightOn",
        "probability": 0.4879843917522865
      },
      "slots": [
        {
          "range": {
            "start": 22,
            "end": 28
          },
          "rawValue": "lounge",
          "value": {
            "kind": "Custom",
            "value": "living room"
          },
          "entity": "room",
          "slotName": "room"
        }
      ]
    }

Notice that the ``lounge`` slot value points to ``living room`` as defined
earlier in the entity synonyms of the dataset.

.. _none_intent:

---------------
The None intent
---------------

On top of the intents that you have declared in your dataset, the NLU engine
generates an implicit intent to cover utterances that does not correspond to
any of your intents. We refer to it as the **None** intent.

The NLU engine is trained to recognize when the input corresponds to the None
intent. Here is what you should get if you try parsing ``"foo bar"`` with the
engine we previously created:

.. code-block:: json

    {
      "input": "foo bar",
      "intent": null,
      "slots": null
    }

Persisting
----------

As a final step, we will persist the engine into a directory. That may be
useful in various contexts, for instance if you want to train on a machine and
parse on another one.

You can persist the engine with the following API:

.. code-block:: python

    engine.persist("path/to/directory")


And load it:

.. code-block:: python

    loaded_engine = SnipsNLUEngine.from_path("path/to/directory")

    loaded_engine.parse(u"Turn lights on in the bathroom please")


Alternatively, you can persist/load the engine as a ``bytearray``:

.. code-block:: python

    engine_bytes = engine.to_byte_array()
    loaded_engine = SnipsNLUEngine.from_byte_array(engine_bytes)


.. _sample dataset: https://github.com/snipsco/snips-nlu/blob/master/snips_nlu_samples/sample_dataset.json
.. _default configurations: https://github.com/snipsco/snips-nlu/blob/master/snips_nlu/default_configs
.. _english one: https://github.com/snipsco/snips-nlu/blob/master/snips_nlu/default_configs/config_en.py
.. _chatito: https://github.com/rodrigopivi/Chatito
