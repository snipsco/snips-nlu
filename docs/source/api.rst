.. _api:

API reference
=============

This part of the documentation covers the most important interfaces of the
Snips NLU package.

Resources
---------

.. automodule:: snips_nlu.resources
   :members:


NLU engine
----------

.. module:: snips_nlu.nlu_engine.nlu_engine

.. autoclass:: SnipsNLUEngine
   :members:


Intent Parser
-------------

.. module:: snips_nlu.intent_parser

.. autoclass:: IntentParser
   :members:

.. autoclass:: DeterministicIntentParser
   :members:

.. autoclass:: ProbabilisticIntentParser
   :members:


Intent Classifier
-----------------

.. module:: snips_nlu.intent_classifier

.. autoclass:: IntentClassifier
   :members:

.. autoclass:: LogRegIntentClassifier
   :members:


Slot Filler
-----------

.. module:: snips_nlu.slot_filler

.. autoclass:: SlotFiller
   :members:

.. autoclass:: CRFSlotFiller
   :members:


-------
Feature
-------

.. autoclass:: Feature
   :members:

-----------------
Feature Factories
-----------------

.. automodule:: snips_nlu.slot_filler.feature_factory
   :members:


Configurations
--------------

.. module:: snips_nlu.pipeline.configs

.. autoclass:: NLUEngineConfig
   :members:

.. autoclass:: DeterministicIntentParserConfig
   :members:

.. autoclass:: ProbabilisticIntentParserConfig
   :members:

.. autoclass:: LogRegIntentClassifierConfig
   :members:

.. autoclass:: CRFSlotFillerConfig
   :members:


Result and output format
------------------------

.. automodule:: snips_nlu.result