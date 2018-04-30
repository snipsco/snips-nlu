.. _data_model:

Key Concepts & Data Model
=========================

This section is meant to explain the concepts and data model that we use to
represent input and output data.

The main task that this lib performs is *Information Extraction*, or *Intent Parsing*, to be even more specific. At this point, the output of the engine may still not be very clear to you.

The task of parsing intents is actually two-folds. The first step is to
understand which intent the sentence is about. The second step is
to extract the parameters, a.k.a. the *slots* of the sentence.

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

They both express the same intent which is **switchLightOn**, but they
are expressed in two very different ways.

Thus, the first task in intent parsing is to be able to detect the *intent* of
the sentence, or say differently to classify sentences based on their
underlying *intent*.

In Snips NLU, this is represented within the parsing output in this way:

.. code-block:: json

    {
        "intentName": "switchLightOn",
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

As before the intent is **switchLightOn**, however there is now an
additional piece of information which is contained in the word **kitchen**.

This intent contains one slot, which is the *room* in which the light is to be
turned on.

Let's consider another example:

.. code-block:: json

    "Find me a flight from Paris to Tokyo"

Here the intent would be **searchFlight**, and now there are two slots in the
sentence being contained in ``"Paris"`` and ``"Tokyo"``. These two values are
of the same type as they both correspond to a **location** however they have
different roles, as Paris is the **departure** and Tokyo is the **arrival**.

In this context, we call **location** a *slot type* (or *entity*) and
**departure** and **arrival** are *slot names*.

.. note::

    We may refer equally to *slot type* or *entity* to describe the same
    concept

.. _entity_vs_slot_name:

-----------------------
Slot type vs. slot name
-----------------------

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

In this example, the slot value contains a ``"kind"`` attribute whose value
here is ``"Custom"``. There are two classes of slot types or entity:

-   **Builtin entities**
-   **Custom entities**


.. _builtin_entity_resolution:

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
      "slotName": "weatherDate"
    }

As you can see, the ``"value"`` field here contains more information than in
the previous example. This is because the entity used here,
``"snips/datetime"``, is what we call a **Builtin Entity**.

Snips NLU supports multiple builtin entities that are typically strongly typed
entities such as date, temperatures, numbers etc, and for which a specific
extractor is available.

These entities have special labels starting with ``"snips/"`` and making use
of them when appropriate will not only give better results, but it will also
provide some *entity resolution* such as an ISO format for a date.

Builtin entities and their underlying extractors are maintained by the Snips
team. You can find the list of all the builtin entities supported per language
in the `Snips NLU Ontology <https://github.com/snipsco/snips-nlu-ontology>`_
repository. The Snips NLU uses the powerful
`Rustling <https://github.com/snipsco/rustling-ontology>`_ library to extract
builtin entities from text.

On the other hand, entities that are declared by the developer are called
*custom* entities.

Custom Entities
---------------

As soon as you use a slot type which is not part of Snips builtin entities, you
are using a custom entity. There are several things you can do to customize it,
and make it fit with your use case.

.. _synonyms:

------------------------
Entity Values & Synonyms
------------------------

The first thing you can do is add a list of possible values for your entity.

By providing a list of example values for your entity, you help Snips NLU
grasp what the entity is about.

Let's say you are creating an assistant whose purpose is to let you set the
color of your connected light bulbs. What you will do is define a ``"color"``
entity. On top of that you can provide a list of sample colors by editing the
entity in your dataset as follows:

.. code-block:: json

    {
      "color": {
        "automatically_extensible": true,
        "use_synonyms": true,
        "data": [
          {
            "value": "white",
            "synonyms": []
          },
          {
            "value": "yellow",
            "synonyms": []
          },
          {
            "value": "pink",
            "synonyms": []
          },
          {
            "value": "blue",
            "synonyms": []
          }
        ]
      }
    }

Now imagine that you want to allow some variations around these values e.g.
using ``"pinky"`` instead of ``"pink"``. You could add these variations in the
list by adding a new value, however in this case what you want is to tell the
NLU to consider ``"pinky"`` as a *synonym* of ``"pink"``:

.. code-block:: json

    {
      "value": "pink",
      "synonyms": ["pinky"]
    }

In this context, Snips NLU will map ``"pinky"`` to its reference value,
``"pink"``, in its output.

Let's consider this sentence:

.. code-block:: console

    Please make the light pinky

Here is the kind of NLU output that you would get in this context:

.. code-block:: json

    {
      "input": "Please make the light pinky",
      "intent": {
        "intentName": "setLightColor",
        "probability": 0.95
      },
      "slots": [
        {
          "rawValue": "pinky",
          "value": {
            "kind": "Custom",
            "value": "pink"
          },
          "entity": "color",
          "slotName": "lightColor",
          "range": {
            "start": 22,
            "end": 27
          }
        }
      ]
    }

The ``"rawValue"`` field contains the color value as written within the input,
but now the ``"value"`` field has been *resolved* and it contains the reference
color, ``"pink"``, that the synonym refers to.


.. _auto_extensible:

---------------------------------
Automatically Extensible Entities
---------------------------------

On top of declaring color values and color synonyms, you can also decide how
Snips NLU reacts to unknown entity values.

In the light color assistant example, one of the first thing to do would be
to check what are the colors that are supported by the bulb, for instance:

.. code-block:: json

    ["white", "yellow", "red", "blue", "green", "pink", "purple"]

As you can only handle these colors, you can enforce Snips NLU to
**filter out slot values that are not part of this list**, so that the output
always contain valid values, i.e. supported colors.

On the contrary, let's say you want to build a smart music assistant that will
let you control your speakers and play any artist you want.

Obviously, you can't list all the artist and songs that you might want to
listen to at some point. This means that your dataset will contain some
examples of such artist but you expect Snips NLU to **extend beyond these values**
and extract any other artist or song that appear in the same context.

Your entity must be *automatically extensible*.

Now in practice, there is a flag in the dataset that lets you choose whether or
not your custom entity is automatically extensible:

.. code-block:: json

    {
      "my_custom_entity": {
        "automatically_extensible": true,
        "use_synonyms": true,
        "data": []
      }
    }
