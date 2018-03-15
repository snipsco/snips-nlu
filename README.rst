Snips NLU
=========

.. image:: https://travis-ci.org/snipsco/snips-nlu.svg?branch=develop
    :target: https://travis-ci.org/snipsco/snips-nlu

.. image:: https://img.shields.io/pypi/v/snips-nlu.svg?branch=develop
    :target: https://pypi.python.org/pypi/snips-nlu

.. image:: https://img.shields.io/pypi/pyversions/snips-nlu.svg?branch=develop
    :target: https://pypi.python.org/pypi/snips-nlu

.. image:: https://codecov.io/gh/snipsco/snips-nlu/branch/develop/graph/badge.svg
   :target: https://codecov.io/gh/snipsco/snips-nlu


`Snips NLU <https://snips-nlu.readthedocs.io>`_ (Natural Language Understanding) is a Python library that allows to parse sentences written in natural language and extracts structured information.


Installing
----------

.. code-block:: python

    pip install snips-nlu

.. note::

   Currently we have built binaries (wheels) for ``snips-nlu`` and its
   dependencies for MacOS and Linux x86_64. If you use different
   architectures/os you will need to build these dependencies from sources
   which means you will need to install
   `setuptools_rust <https://github.com/PyO3/setuptools-rust>`_ and
   `Rust <https://www.rust-lang.org/en-US/install.html>`_.


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

To find out how to use Snips NLU please refer to our `documentation <https://snips-nlu.readthedocs.io>`_, it will provide you with a step-by-step guide on how to use and setup our library.


Links
-----
* `Snips NLU <https://github.com/snipsco/snips-nlu>`_
* `Snips NLU Rust <https://github.com/snipsco/snips-nlu-rs>`_: Rust inference pipeline implementation and bindings (C, Swift, Kotlin, Python)
* `Rustling <https://github.com/snipsco/rustling-ontology>`_: Snips NLU builtin entities parser
* `Snips <https://snips.ai/>`_
* `Bug tracker <https://github.com/snipsco/snips-nlu/issues>`_
