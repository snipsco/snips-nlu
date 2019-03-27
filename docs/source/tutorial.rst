.. _tutorial:

Tutorial
========

In this section, we will build an NLU assistant for home automation tasks. It
will be able to understand queries about lights and thermostats. More
precisely, our assistant will contain three :ref:`intents <intent>`:

- ``turnLightOn``
- ``turnLightOff``
- ``setTemperature``

The first two intents will be about turning on and off the lights in a specific
room. These intents will have one :ref:`slot` which will be the ``room``.
The third intent will let you control the temperature of a specific room. It
will have two slots: the ``roomTemperature`` and the ``room``.

The first step is to create an appropriate dataset for this task.

Training Data
-------------

Check the :ref:`Training Dataset Format <dataset>` section for more details
about the format used to describe the training data.

In this tutorial, we will create our dataset using the
:ref:`YAML format <yaml_format>`, and create a ``dataset.yaml`` file with the
following content:

.. code-block:: yaml

    # turnLightOn intent
    ---
    type: intent
    name: turnLightOn
    slots:
      - name: room
        entity: room
    utterances:
      - Turn on the lights in the [room](kitchen)
      - give me some light in the [room](bathroom) please
      - Can you light up the [room](living room) ?
      - switch the [room](bedroom)'s lights on please

    # turnLightOff intent
    ---
    type: intent
    name: turnLightOff
    slots:
      - name: room
        entity: room
    utterances:
      - Turn off the lights in the [room](entrance)
      - turn the [room](bathroom)'s light out please
      - switch off the light the [room](kitchen), will you?
      - Switch the [room](bedroom)'s lights off please

    # setTemperature intent
    ---
    type: intent
    name: setTemperature
    slots:
      - name: room
        entity: room
      - name: roomTemperature
        entity: snips/temperature
    utterances:
      - Set the temperature to [roomTemperature](19 degrees) in the [room](bedroom)
      - please set the [room](living room)'s temperature to [roomTemperature](twenty two degrees celsius)
      - I want [roomTemperature](75 degrees fahrenheit) in the [room](bathroom) please
      - Can you increase the temperature to [roomTemperature](22 degrees) ?

    # room entity
    ---
    type: entity
    name: room
    automatically_extensible: no
    values:
    - bedroom
    - [living room, main room, lounge]
    - [garden, yard, backyard]

Here, we put all the intents and entities in the same file but we could have
split them in dedicated files as well.

The ``setTemperature`` intent references a ``roomTemperature`` slot which
relies on the ``snips/temperature`` entity. This entity is a
:ref:`builtin entity <builtin_entity_resolution>`. It allows to resolve the
temperature values properly.

The ``room`` entity makes use of :ref:`synonyms <synonyms>` by defining lists
like ``[living room, main room, lounge]``. In this case, ``main room`` and
``lounge`` will point to ``living room``, the first item of the list, which is
the reference value.

Besides, this entity is marked as not
:ref:`automatically extensible <auto_extensible>` which means that the NLU
will only output values that we have defined and will not try to match other
values.

We are now ready to generate our dataset using the :ref:`CLI <cli>`:

.. code-block:: bash

    snips-nlu generate-dataset en dataset.yaml > dataset.json

.. note::

    We used ``en`` as the language here but other languages are supported,
    please check the :ref:`languages` section to know more.

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

.. code-block:: python

    import io
    import json

    from snips_nlu import SnipsNLUEngine
    from snips_nlu.default_configs import CONFIG_EN

    engine = SnipsNLUEngine(config=CONFIG_EN)

At this point, we can try to parse something:

.. code-block:: python

    engine.parse(u"Please give me some lights in the entrance !")

That will raise a ``NotTrained`` error, as we did not train the engine with
the dataset that we created.


.. _training_the_engine:

Training the engine
-------------------

In order to use the engine we created, we need to *train* it or *fit* it with
the dataset we generated earlier:

.. code-block:: python

    with io.open("dataset.json") as f:
        dataset = json.load(f)

    engine.fit(dataset)

Note that the training of the engine is not deterministic by default. This
means that if you train your NLU twice on the same data you'll end up
the output of the NLU can be different.

To


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

Now, let's say the intent is already known and provided by the context of the
application, but the slots must still be extracted. A second parsing API allows
to extract the slots while providing the intent:

.. code-block:: python

   parsing = engine.get_slots(u"Hey, lights on in the lounge !", "turnLightOn")
   print(json.dumps(parsing, indent=2))

This will give you only the extracted slots:

.. code-block:: json

   [
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

Finally, there is another method that allows to run only the intent
classification and get the list of intents along with their score:

.. code-block:: python

    intents = engine.get_intents(u"Hey, lights on in the lounge !")
    print(json.dumps(intents, indent=2))

This should give you something like below:

.. code-block:: json

   [
     {
       "intentName": "turnLightOn",
       "probability": 0.6363648460343694
     },
     {
       "intentName": null,
       "probability": 0.2580088944934134
     },
     {
       "intentName": "turnLightOff",
       "probability": 0.22791834836267366
     },
     {
       "intentName": "setTemperature",
       "probability": 0.181781583254962
     }
   ]

You will notice that the second intent is ``null``. This intent is what we
call the :ref:`None intent <none_intent>` and is explained in the next
section.

.. important::

    Even though the term ``"probability"`` is used here, the values should
    rather be considered as confidence scores as they do not sum to 1.0.


.. _none_intent:

---------------
The None intent
---------------

On top of the intents that you have declared in your dataset, the NLU engine
generates an implicit intent to cover utterances that does not correspond to
any of your intents. We refer to it as the **None** intent.

The NLU engine is trained to recognize when the input corresponds to the None
intent. Here is the kind of output you should get if you try parsing
``"foo bar"`` with the engine we previously created:

.. tabs::

   .. tab:: Python

      .. code-block:: python

          {
            "input": "foo bar",
            "intent": {
              "intentName": None,
              "probability": 0.552122
            },
            "slots": []
          }

   .. tab:: JSON

      .. code-block:: json

          {
            "input": "foo bar",
            "intent": {
              "intentName": null,
              "probability": 0.552122
            },
            "slots": []
          }

The **None** intent is represented by a ``None`` value in python which
translates in JSON into a ``null`` value.

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
