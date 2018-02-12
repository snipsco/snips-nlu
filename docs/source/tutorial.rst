.. _tutorial:

Tutorial
========

In this section, we will build an NLU assistant for home automation tasks that
will be able to understand queries about lights and thermostat. More precisely
our assistant will contain two :ref:`intents <intent>`:

- ``TurnLightOn``
- ``TurnLightOff``
- ``SetTemperature``

The first two intents will be about turning on and off the lights in a specific
room. Thus, these intents will have one :ref:`slot` which will be the ``room``.
The third intent will let you control the temperature of a specific room, thus
it will have two slots: the ``room_temperature`` and the ``room``.

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

We will go for the second option here and start by creating two files
corresponding to our two intents:

- ``TurnLightOn.txt``
- ``TurnLightOff.txt``
- ``SetTemperature.txt``

The name of each file is important as the tool will map it to the intent name.

Let's add training examples for the first intent by inserting the following
lines in the first file, ``TurnLightOn.txt``:

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

Let's move on to the second intent, and insert this into ``TurnLightOff.txt``:

.. code-block:: console

    Turn off the lights in the [room:room](entrance)
    turn the [room:room](bathroom)'s light out please
    switch off the light the [room:room](kitchen), will you?
    Switch the [room:room](bedroom)'s lights off please

And now the last file, ``SetTemperature.txt``:

.. code-block:: console

    Set the temperature to [room_temperature:snips/temperature](19 degrees) in the [room:room](bedroom)
    please set the [room:room](living room)'s temperature to [room_temperature:snips/temperature](twenty two degrees celsius)
    I want [room_temperature:snips/temperature](75 degrees fahrenheit) in the [room:room](bathroom) please
    Can you increase the temperature to [room_temperature:snips/temperature](22 degrees) ?

As you can see here, we used a new slot, ``[room_temperature:snips/temperature]``,
which name is ``room_temperature`` and type is ``snips/temperature``. The slot
type that we used here is a :ref:`builtin entity <builtin_entity_resolution>`
that would help us resolve properly the temperature values.

We are now ready to generate our dataset:

.. code-block:: console

    generate-dataset --language en TurnLightOn.txt TurnLightOff.txt SetTemperature.txt > dataset.json



.. _sample dataset: https://github.com/snipsco/snips-nlu/blob/master/samples/sample_dataset.json