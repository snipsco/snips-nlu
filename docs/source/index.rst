Snips Natural Language Understanding
====================================

.. image:: https://travis-ci.org/snipsco/snips-nlu.svg?branch=develop
    :target: https://travis-ci.org/snipsco/snips-nlu

.. image:: https://img.shields.io/pypi/v/snips-nlu.svg?branch=develop
    :target: https://pypi.python.org/pypi/snips-nlu

.. image:: https://img.shields.io/pypi/pyversions/snips-nlu.svg?branch=develop
    :target: https://pypi.python.org/pypi/snips-nlu

.. image:: https://codecov.io/gh/snipsco/snips-nlu/branch/develop/graph/badge.svg
   :target: https://codecov.io/gh/snipsco/snips-nlu

Welcome to Snips NLU's documentation.

Snips NLU is a Natural Language Understanding python library that allows to
parse sentences written in natural language, and extract structured
information.

It's the library that powers the NLU engine used in the
`Snips Console <https://console.snips.ai/>`_ that you can use to create awesome
and private-by-design voice assistants.

Let's look at the following example, to illustrate the main purpose of this lib:

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
            "slotName": "forecastLocality"
         },
         {
            "value": {
               "kind": "InstantTime",
               "value": "2018-02-08 20:00:00 +00:00"
            },
            "entity": "snips/datetime",
            "slotName": "forecastStartDatetime"
         }
      ]
   }

.. note::

   The exact output is a bit richer, the point here is to give a glimpse on
   what kind of information can be extracted.

This documentation is divided into different parts. It is recommended to
start by the first two ones.

The :ref:`installation` part will get you set up. Then, the :ref:`quickstart`
section will help you build a toy example.

After this, you can either start the :ref:`tutorial` which will guide you
through the steps to create your own NLU engine and start parsing sentences, or
you can alternatively check the :ref:`data_model` to know more about the NLU
concepts used in this lib.

If you want to dive into the codebase or customize some parts, you can use
the :ref:`api` documentation or alternatively check the `github repository`_.

.. toctree::
   :maxdepth: 2

   installation
   quickstart
   tutorial
   data_model
   dataset
   custom_processing_units
   languages
   cli
   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _github repository: https://github.com/snipsco/snips-nlu

