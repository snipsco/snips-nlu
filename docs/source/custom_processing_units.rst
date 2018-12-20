.. _custom_processing_units:

Custom Processing Units
=======================

The Snips NLU library provides a default NLU pipeline containing builtin
processing units such as the :class:`.DeterministicIntentParser` or the
:class:`.ProbabilisticIntentParser`.

However, it is possible to define custom processing units and use them in a
:class:`.SnipsNLUEngine`.

The main processing unit of the Snips NLU processing pipeline is the
:class:`.SnipsNLUEngine`. This engine relies on a list of :class:`.IntentParser`
that are called successively until one of them manages to extract an intent.
By default, two parsers are used by the engine: a
:class:`.DeterministicIntentParser` and a :class:`.ProbabilisticIntentParser`.

Let's focus on the probabilistic intent parser. This parser parses text using
two steps: first it classifies the intent using an
:class:`.IntentClassifier` and once the intent is known, it using a
:class:`.SlotFiller` in order to extract the slots.

For the purpose of this tutorial, let's build a custom alternative to the
:class:`.CRFSlotFiller` which is the default slot filler used by the
probabilistic intent parser.

Our custom slot filler will extract slots by relying on a very simple and
naive keyword matching logic:

.. code-block:: python

   import json

   from snips_nlu.common.utils import json_string
   from snips_nlu.preprocessing import tokenize
   from snips_nlu.result import unresolved_slot
   from snips_nlu.slot_filler import SlotFiller


   @SlotFiller.register("keyword_slot_filler")
   class KeywordSlotFiller(SlotFiller):
       def __init__(self, config=None, **shared):
           super(KeywordSlotFiller, self).__init__(config, **shared)
           self.slots_keywords = None
           self.language = None

       @property
       def fitted(self):
           return self.slots_keywords is not None

       def fit(self, dataset, intent):
           self.language = dataset["language"]
           self.slots_keywords = dict()
           utterances = dataset["intents"][intent]["utterances"]
           for utterance in utterances:
               for chunk in utterance["data"]:
                   if "slot_name" in chunk:
                       text = chunk["text"]
                       self.slots_keywords[text] = [
                           chunk["entity"],
                           chunk["slot_name"]
                       ]
           return self

       def get_slots(self, text):
           tokens = tokenize(text, self.language)
           slots = []
           for token in tokens:
               value = token.value
               if value in self.slots_keywords:
                   entity = self.slots_keywords[value][0]
                   slot_name = self.slots_keywords[value][1]
                   slot = unresolved_slot((token.start, token.end), value,
                                          entity, slot_name)
                   slots.append(slot)
           return slots

       def persist(self, path):
           model = {
               "language": self.language,
               "slots_keywords": self.slots_keywords,
               "config": self.config.to_dict()
           }
           with path.open(mode="w") as f:
               f.write(json_string(model))

       @classmethod
       def from_path(cls, path, **shared):
           with path.open() as f:
               model = json.load(f)
           slot_filler = cls()
           slot_filler.language = model["language"]
           slot_filler.slots_keywords = model["slots_keywords"]
           slot_filler.config = cls.config_type.from_dict(model["config"])
           return slot_filler

Our custom slot filler is registered to the list of available processing units
by the use of a class decorator:
``@SlotFiller.register("keyword_slot_filler")``.

Now that we have created our keyword slot filler, we can create a specific
:class:`NLUEngineConfig` which will make use of it:

.. code-block:: python

   from snips_nlu import SnipsNLUEngine
   from snips_nlu.pipeline.configs import (
       ProbabilisticIntentParserConfig, NLUEngineConfig)
   from snips_nlu.slot_filler.keyword_slot_filler import KeywordSlotFiller

   slot_filler_config = KeywordSlotFiller.default_config()
   parser_config = ProbabilisticIntentParserConfig(
       slot_filler_config=slot_filler_config)
   engine_config = NLUEngineConfig([parser_config])
   nlu_engine = SnipsNLUEngine(engine_config)


Custom processing unit configuration
------------------------------------

So far, our keyword slot filler is very simple, especially because it is not
configurable.

Now, let's imagine that we would like to perform a normalization step
before matching keywords, which would consist in lowercasing the values.
We could hardcode this behavior in our unit, but what would be ideal is to have
a way to configure this behavior. This can be done through the use of the
``config`` attribute of our keyword slot filler. Let's add a boolean parameter
in the config, so that now our ``KeywordSlotFiller`` implementation looks like
this:

.. code-block:: python

   import json

   from snips_nlu.common.utils import json_string
   from snips_nlu.preprocessing import tokenize
   from snips_nlu.result import unresolved_slot
   from snips_nlu.slot_filler import SlotFiller


   @SlotFiller.register("keyword_slot_filler")
   class KeywordSlotFiller(SlotFiller):
       def __init__(self, config=None, **shared):
           super(KeywordSlotFiller, self).__init__(config, **shared)
           self.slots_keywords = None
           self.language = None

       @property
       def fitted(self):
           return self.slots_keywords is not None

       def fit(self, dataset, intent):
           self.language = dataset["language"]
           self.slots_keywords = dict()
           utterances = dataset["intents"][intent]["utterances"]
           for utterance in utterances:
               for chunk in utterance["data"]:
                   if "slot_name" in chunk:
                       text = chunk["text"]
                       if self.config.get("lowercase", False):
                           text = text.lower()
                       self.slots_keywords[text] = [
                           chunk["entity"],
                           chunk["slot_name"]
                       ]
           return self

       def get_slots(self, text):
           tokens = tokenize(text, self.language)
           slots = []
           for token in tokens:
               normalized_value = token.value
               if self.config.get("lowercase", False):
                   normalized_value = normalized_value.lower()
               if normalized_value in self.slots_keywords:
                   entity = self.slots_keywords[normalized_value][0]
                   slot_name = self.slots_keywords[normalized_value][1]
                   slot = unresolved_slot((token.start, token.end), token.value,
                                          entity, slot_name)
                   slots.append(slot)
           return slots

       def persist(self, path):
           model = {
               "language": self.language,
               "slots_keywords": self.slots_keywords,
               "config": self.config.to_dict()
           }
           with path.open(mode="w") as f:
               f.write(json_string(model))

       @classmethod
       def from_path(cls, path, **shared):
           with path.open() as f:
               model = json.load(f)
           slot_filler = cls()
           slot_filler.language = model["language"]
           slot_filler.slots_keywords = model["slots_keywords"]
           slot_filler.config = cls.config_type.from_dict(model["config"])
           return slot_filler

Now we can define a more specific config for our slot filler:

.. code-block:: python

   from snips_nlu import SnipsNLUEngine
   from snips_nlu.pipeline.configs import (
       ProbabilisticIntentParserConfig, NLUEngineConfig)
   from snips_nlu.slot_filler.keyword_slot_filler import KeywordSlotFiller

   slot_filler_config = {
       "unit_name": "keyword_slot_filler",  # required in order to identify the processing unit
       "lower_case": True
   }
   parser_config = ProbabilisticIntentParserConfig(
       slot_filler_config=slot_filler_config)
   engine_config = NLUEngineConfig([parser_config])
   nlu_engine = SnipsNLUEngine(engine_config)


You can now use train this engine, parse intents, persist it and load it from
disk.

.. note::

    The client code is responsible for persisting and loading the unit
    configuration as done in the implementation example. This will ensure
    that the proper configuration is used when deserializing the processing
    unit.
