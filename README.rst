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

Check out our `blog post`_ to get more details about why we built Snips NLU and how it works under the hood.

Installation
------------

.. code-block:: python

    pip install snips-nlu

We currently have pre-built binaries (wheels) for ``snips-nlu`` and its
dependencies for MacOS and Linux x86_64. If you use a different
architecture/os you will need to build these dependencies from sources
which means you will need to install
`setuptools_rust <https://github.com/PyO3/setuptools-rust>`_ and
`Rust <https://www.rust-lang.org/en-US/install.html>`_ before running the
``pip install snips-nlu`` command.

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


Sample code
-----------

Here is a sample code that you can run on your machine after having
installed `snips-nlu` and downloaded this `sample dataset`_:

.. code-block:: python

    from __future__ import unicode_literals, print_function

    import io
    import json

    from snips_nlu import SnipsNLUEngine, load_resources
    from snips_nlu.default_configs import CONFIG_EN

    with io.open("sample_dataset.json") as f:
        sample_dataset = json.load(f)

    load_resources("en")
    nlu_engine = SnipsNLUEngine(config=CONFIG_EN)
    nlu_engine.fit(sample_dataset)

    text = "What will be the weather in San Francisco next week?"
    parsing = nlu_engine.parse(text)
    print(json.dumps(parsing, indent=2))

What it does is training an NLU engine on a sample weather dataset and parsing
a weather query.

Documentation
-------------

To find out how to use Snips NLU please refer to our `documentation <https://snips-nlu.readthedocs.io>`_, it will provide you with a step-by-step guide on how to use and setup our library.

FAQ
---
Please join our `Discord channel`_ to ask your questions and get feedback from the community.

Links
-----
* `What is Snips about ? <https://snips.ai/>`_
* Snips NLU Open sourcing `blog post`_
* `Bug tracker <https://github.com/snipsco/snips-nlu/issues>`_
* `Snips NLU Rust <https://github.com/snipsco/snips-nlu-rs>`_: Rust inference pipeline implementation and bindings (C, Swift, Kotlin, Python)
* `Rustling <https://github.com/snipsco/rustling-ontology>`_: Snips NLU builtin entities parser


How do I contribute ?
---------------------

Please see the `Contribution Guidelines <CONTRIBUTING.rst>`_.

Licence
-------

This library is provided by `Snips <https://www.snips.ai>`_ as Open Source software. See `LICENSE <LICENSE>`_ for more information.

.. _sample dataset: samples/sample_dataset.json
.. _Discord channel: https://discordapp.com/invite/3939Kqx
.. _blog post: https://medium.com/snips-ai/an-introduction-to-snips-nlu-the-open-source-library-behind-snips-embedded-voice-platform-b12b1a60a41a