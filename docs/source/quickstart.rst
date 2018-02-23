.. _quickstart:

Quickstart
==========

The Snips NLU engine, in its default configuration, needs to be trained on
some data before it can start extracting information. Thus, the first thing to
do is to build a dataset that can be fed into Snips NLU.
For now, we will use this `sample dataset`_ which contains data for two intents:

- ``sampleGetWeather`` -> ``"What will be the weather in Tokyo tomorrow?"``
- ``sampleTurnOnLight`` -> ``"Turn on the light in the kitchen"``

The format used here is json so let's load it into a python dict:

.. code-block:: python

    import io
    import json

    with io.open("path/to/sample_dataset.json") as f:
        sample_dataset = json.load(f)

Now that we have our dataset, we can move forward to the next step which is
building a :class:`.SnipsNLUEngine` which is the main object of this lib.
Before training the engine, note that you need to load language specific
resources used to improve performance with the :func:`.load_resources` function.

.. code-block:: python

    from snips_nlu import load_resources, SnipsNLUEngine

    load_resources(u"en")
    nlu_engine = SnipsNLUEngine()

Now that we have our engine object created, we need to feed it with our sample
dataset. In general, this action will require some *machine learning* hence we
will actually *fit* the engine:

.. code-block:: python

    nlu_engine.fit(sample_dataset)


Our NLU engine is now trained to recognize new utterances that extend beyond
what is strictly contained in the dataset, it is able to *generalize*.

Let's try to parse something now!

.. code-block:: python

    import json

    parsing = nlu_engine.parse(u"What will be the weather in San Francisco next week?")
    print(json.dumps(parsing, indent=2))


You should get something that looks like this:

.. code-block:: json

    {
      "input": "What will be the weather in San Francisco next week?",
      "intent": {
        "intentName": "sampleGetWeather",
        "probability": 0.641227710154331
      },
      "slots": [
        {
          "range": {
            "start": 28,
            "end": 41
          },
          "rawValue": "San Francisco",
          "value": {
            "kind": "Custom",
            "value": "San Francisco"
          },
          "entity": "location",
          "slotName": "weatherLocation"
        },
        {
          "range": {
            "start": 42,
            "end": 51
          },
          "rawValue": "next week",
          "value": {
            "type": "value",
            "grain": "week",
            "precision": "exact",
            "latent": false,
            "value": "2018-02-12 00:00:00 +01:00"
          },
          "entity": "snips/datetime",
          "slotName": "weatherDate"
        }
      ]
    }

Congrats, you parsed your first intent!


.. _sample dataset: https://github.com/snipsco/snips-nlu/blob/master/samples/sample_dataset.json