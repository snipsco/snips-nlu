.. _data_model:


Snips NLU Concepts & Data Model
===============================

This section is meant to explain the concepts and data model that we use to
represent input and output data.

The main task that this lib performs falls into a category of tasks that are
called *Information Extraction* tasks. In our case, the task is more
specifically called *Intent Parsing*. At this point, the output may still not
be very clear.

The task of parsing intents is actually two folds, as the first step is to
understand what kind of intent the sentence is about, and the second step is
to extract the parameters, aka the *slots*, of the sentence.

.. _intent:

Intent
------

In the context of information extraction, an *intent* corresponds to the
action or intention contained in the user's query, which can be more or less
explicit.

Lets' consider for instance the following sentences:

.. code-block:: json

    "Turn on the light"
    "It's too dark in this room, can you fix this?"

They both express the same intent which is **Switch Light On**, but they
are expressed in two very different ways.

Thus, the first task in intent parsing is to be able to detect the *intent* of
the sentence, or say differently to classify sentences based on their
underlying *intent*.

In Snips NLU, this is represented within the parsing output in this way:

.. code-block:: json

    {
        "intentName": "Switch Light On",
        "probability": 0.87
    }

So you have an additional information which is the probability that the
extracted intent correspond to the actual one.


.. _slot:

Slot
----

The second part of the task, once the intent is known, is to extract the
parameters that may be contained in the sentence. We called them *slots*.

For example, let's consider this sentence:

.. code-block:: json

    "Turn on the light in the kitchen"

As before the intent is **Switch Light On**, however there is now an
additional piece of information which is contained in the word **kitchen**.

This intent contains one slot, which is the *room* in which the light is to be
turned on.

Let's consider another example:

.. code-block:: json

    "Find me a flight from Paris to Tokyo"

Here the intent would be **Search Flight**, and now there are two slots in the
sentence being contained in ``"Paris"`` and ``"Tokyo"``. These two values are
of the same type as they both correspond to a **location** however they have
different roles, as Paris is the **departure** and Tokyo is the **arrival**.

In this context, we call **location** a *slot type* (or *entity*) and
**departure** and **arrival** are *slot names*.

.. note::

    We may refer equally to *slot type* or *entity* to describe the same
    concept

.. _entity_vs_slot_name:

----------------------
Slot type VS slot name
----------------------

A slot type or entity is to NLU what a type is to coding. It describes the
nature of the value. In a piece of code, multiple variables can be of the same
type while having different purposes, usually transcribed in their name. All
variables of a same type will have some common characteristics, for instance
they have the same methods, they may be comparable etc.

In information extraction, a slot type corresponds to a class of values that
fall into the same category. In our previous example, the **location** slot
type corresponds to all values that correspond to a place, a city, a country or
anything that can be located.

The slot name can be thought as the *role* played by the entity in the
sentence.


In Snips NLU, extracted slots are represented within the output in this way:

.. code-block:: json

    [
      {
        "rawValue": "Paris",
        "value": {
          "kind": "Custom",
          "value": "Paris"
        },
        "entity": "location",
        "slotName": "departure",
        "range": {
          "start": 28,
          "end": 41
        }
      },
      {
        "rawValue": "Tokyo",
        "value": {
          "kind": "Custom",
          "value": "Tokyo"
        },
        "entity": "location",
        "slotName": "arrival",
        "range": {
          "start": 28,
          "end": 41
        }
      }
    ]

.. _builtin_entity_resolution:

-------------------------------
Builtin Entities and resolution
-------------------------------

Snips NLU actually goes a bit further than simply extracting slots, let's
illustrate this with another example:

.. code-block:: json

    "What will be the weather tomorrow at 10am?"

This sentence contains a slot, ``"tomorrow at 10am"``, which is a datetime.
Here is how the slot extracted by Snips NLU would look like in this case:

.. code-block:: json

    {
      "rawValue": "tomorrow at 10am",
      "value": {
        "kind": "InstantTime",
        "value": "2018-02-10 10:00:00 +00:00",
        "grain": "Hour",
        "precision": "Exact"
      },
      "range": {
        "start": 20,
        "end": 36
      },
      "entity": "snips/datetime",
      "slotName": "weather_date"
    }

As you can see, the ``"value"`` field here contains more information than in
the previous example. This is because the entity used here,
``"snips/datetime"``, is what we call a **builtin entity**.

Snips NLU supports multiple builtin entities that are typically strongly typed
entities such as date, temperatures, numbers etc, and for which a specific
extractor is available.

These entities have special labels starting with ``"snips/"`` and making use
of them when appropriate will not only give better results, but it will also
provide some *entity resolution* such as an ISO format for a date.