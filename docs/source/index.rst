Snips Natural Language Understanding
====================================

Welcome to Snips NLU's documentation.

This tool is a Natural Language Understanding python library that allows to
parse sentences written in natural language and extracts structured
information.

Let's take an example to illustrate the main purpose of this lib, and consider
the following sentence:

.. code-block:: console

   "What will be the weather in paris at 9pm?"

Properly trained, the Snips NLU engine will be able to extract structured data
such as:

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
               "value": "2018-02-08 20:00:00 +00:00",
            },
            "entity": "snips/datetime",
            "slotName": "forecast_start_datetime"
         }
      ]
   }

.. note::

   The exact output is a bit richer, the point here is to give a glimpse on
   what kind of information can be extracted.

This documentation is divided into different parts, it is recommended to
start by the first two ones.

The :ref:`installation` part will get you setup then the :ref:`quickstart`
section will help you build a toy example.
After this, you can either start the :ref:`tutorial` which will guide you
through the steps to create your own NLU engine and start parsing sentences or
alternatively you can check the :ref:`data_model` to know more about the NLU
concepts used in this lib.

If you want to dive into the codebase or customize some parts, you can use
the :ref:`api` documentation or alternatively check the `github repository`_.

.. toctree::
   :maxdepth: 2

   installation
   quickstart
   tutorial
   data_model
   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _github repository: https://github.com/snipsco/snips-nlu

