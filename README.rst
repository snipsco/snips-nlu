Snips NLU
=========


The `Snips NLU <https://snips-nlu.readthedocs.io>`_ (Natural Language Understanding) is a Python library that allows to parse sentences written in natural language and extracts structured information.


Installing
----------

.. code-block:: python

    pip install snips-nlu


A simple example
----------------

Let’s take an example to illustrate the main purpose of this lib, and consider the following sentence:

.. code-block:: text

    "What will be the weather in paris at 9pm?"

Properly trained, the Snips NLU engine will be able to extract structured data such as:

.. code-block:: json

    {
       "intent": {
          "intentName": "searchWeatherForecast",
          "probability": 0.95
       },
       "slots": [
          {
             "value": "paris",
             "entity": "locality",
             "slotName": "forecast_locality"
          },
          {
             "value": {
                "kind": "InstantTime",
                "value": "2018-02-08 20:00:00 +00:00"
             },
             "entity": "snips/datetime",
             "slotName": "forecast_start_datetime"
          }
       ]
    }


Documentation
-------------

To find out how to use the Snips NLU please refer to our `documentation <https://snips-nlu.readthedocs.io>`_, it will provide you with a step-by-step guide on how to use and setup our library.


Links
-----
* `Snips NLU <https://github.com/snipsco/snips-nlu>`_
* `Snips <https://snips.ai/>`_
* `Rustling <https://github.com/snipsco/rustling-ontology>`_ (Snips NLU builtin entities parser)
* `Bug tracker <https://github.com/snipsco/snips-nlu/issues>`_
