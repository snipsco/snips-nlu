.. _tutorial:

Tutorial
========

In this section, we will build an NLU assistant for home automation tasks that
will be able to understand queries about lights and thermostat. More precisely
our assistant will contain two :ref:`intents <intent>`:

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

You have two options to create your dataset. You can build it manually by
respecting the format used in the sample or alternatively you can use the
dataset creation CLI that is contained in the lib.

We will go for the second option here and start by creating three files
corresponding to our three intents and one entity file corresponding to the ``room`` entity:

- ``turnLightOn.txt``
- ``turnLightOff.txt``
- ``setTemperature.txt``
- ``room.txt``

The name of each file is important as the tool will map it to the intent or entity name.

Let's add training examples for the first intent by inserting the following
lines in the first file, ``turnLightOn.txt``:

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

Let's move on to the second intent, and insert this into ``turnLightOff.txt``:

.. code-block:: console

    Turn off the lights in the [room:room](entrance)
    turn the [room:room](bathroom)'s light out please
    switch off the light the [room:room](kitchen), will you?
    Switch the [room:room](bedroom)'s lights off please

And now the last file, ``setTemperature.txt``:

.. code-block:: console

    Set the temperature to [roomTemperature:snips/temperature](19 degrees) in the [room:room](bedroom)
    please set the [room:room](living room)'s temperature to [roomTemperature:snips/temperature](twenty two degrees celsius)
    I want [roomTemperature:snips/temperature](75 degrees fahrenheit) in the [room:room](bathroom) please
    Can you increase the temperature to [roomTemperature:snips/temperature](22 degrees) ?

As you can see here, we used a new slot, ``[room_temperature:snips/temperature]``,
which name is ``roomTemperature`` and type is ``snips/temperature``. The slot
type that we used here is a :ref:`builtin entity <builtin_entity_resolution>`
that would help us resolve properly the temperature values.

Let's move to the ``room.txt`` entity file:

.. code-block:: console

    bedroom
    living room,main room
    garden,yard,"backyard,"

The entity file is a comma (``,``) separated file. Each line correspond to a entity value followed by its potential :ref:`synonyms <synonyms>`.

If a value or a synonym has a comma in it, the value must be put between double quotes ``"``, if the value contains double quotes, it must be doubled to be escaped like this:  ``"A value with a "","" in it"`` which correspond to the actual value ``A value with a "," in it``

We are now ready to generate our dataset:

.. code-block:: bash

    generate-dataset --language en --intent-files   turnLightOn.txt turnLightOff.txt setTemperature.txt --entity-files room.txt > dataset.json

.. note::

    We used ``en`` as the language here but other languages are supported,
    please check the :ref:`languages` section to know more.

Let's have a look at what has been generated and more precisely the
``"entities"`` part of the json:

.. code-block:: json

    {
      "entities": {
        "room": {
          "use_synonyms": true,
          "automatically_extensible": true,
          "data": [
            {
              "value": "bedroom",
              "synonyms": []
            },
            {
              "value": "living room",
              "synonyms": ["main room"]
            },
            {
              "value": "bathroom",
              "synonyms": []
            },
            {
              "value": "garden",
              "synonyms": ["yard", "backyard,"]
            }
          ]
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
Moreover we are going to add some rooms that were not in the previous sentences
and that we want our assistant to cover. We also add some
:ref:`synonyms <synonyms>`, so at the end this is what we have:

.. code-block:: json

    {
      "entities": {
        "room": {
          "use_synonyms": true,
          "automatically_extensible": false,
          "data": [
            {
              "value": "bedroom",
              "synonyms": ["sleeping room"]
            },
            {
              "value": "living room",
              "synonyms": ["main room"]
            },
            {
              "value": "bathroom",
              "synonyms": []
            },
            {
              "value": "garden",
              "synonyms": ["yard", "backyard,"]
            }
          ]
        },
        "snips/temperature": {}
      }
    }

We don't need to edit the ``snips/temperature`` entity as it is a builtin entity.

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

However, in some cases it may be required to tune a bit the engine and provide
a customized configuration. Typically, different languages may require
different sets of features. You can check the :class:`.NLUEngineConfig` to get
more details about what can be configured.

We created a list of `sample configurations`_, one per supported language, that
have some language specific enhancements. In this tutorial we will use the
`english one`_.

.. code-block:: python

    import io
    import json

    from snips_nlu import SnipsNLUEngine

    with io.open("config_en.json") as f:
        config = json.load(f)

    engine = SnipsNLUEngine(config=config)

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

    parsing = engine.parse(u"Hey, lights on in the entrance !")
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

Persisting
----------

As a final step, we will persist the engine in a json. That may be useful in
various contexts, for instance if you want to train on a machine and parse on
another one.

You can persist the engine with the following API:

.. code-block:: python

    engine_json = json.dumps(engine.to_dict())
    with io.open("trained_engine.json", mode="w") as f:
        # f.write(engine_json.decode("utf8"))  # Python 2
        f.write(engine_json)  # Python 3


And load it:

.. code-block:: python


    with io.open("trained_engine.json") as f:
        engine_dict = json.load(f)

    loaded_engine = SnipsNLUEngine.from_dict(engine_dict)

    loaded_engine.parse(u"Turn lights on in the bathroom please")



.. _sample dataset: https://github.com/snipsco/snips-nlu/blob/master/samples/sample_dataset.json
.. _sample configurations: https://github.com/snipsco/snips-nlu/blob/master/samples/configs
.. _english one: https://github.com/snipsco/snips-nlu/blob/master/samples/configs/config_en.json