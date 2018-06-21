# coding=utf-8
from __future__ import unicode_literals, print_function

from pathlib import Path

from snips_nlu.cli.dataset.entities import CustomEntity, create_entity
from snips_nlu.cli.dataset.intent_dataset import IntentDataset


class AssistantDataset(object):
    """Dataset of an assistant

    Merges a list of :class:`.AssistantDataset` into a single dataset ready to
    be used by Snips NLU

    Attributes:
        language (str): language of the assistant
        intents_datasets (list of :class:`.IntentDataset`): data of the
            assistant intents
        entities (list of :class:`.Entity`): data of the assistant entities
    """

    def __init__(self, language, intent_datasets, entities):
        self.language = language
        self.intents_datasets = intent_datasets
        self.entities = entities

    @classmethod
    def from_files(cls, language, filenames):
        """Creates an :class:`.AssistantDataset` from a language and a list of
        intent and entity files

        Args:
            language (str): language of the assistant
            filenames (list of str): Intent and entity files.
                The assistant will associate each intent file to an intent,
                and each entity file to an entity. For instance, the intent
                file 'intent_setTemperature.txt' will correspond to the intent
                'setTemperature', and the entity file 'entity_room.txt' will
                correspond to the entity 'room'.
        """
        intent_filepaths = set()
        entity_filepaths = set()
        for filename in filenames:
            filepath = Path(filename)
            stem = filepath.stem
            if stem.startswith("intent_"):
                intent_filepaths.add(filepath)
            elif stem.startswith("entity_"):
                entity_filepaths.add(filepath)
            else:
                raise AssertionError("Filename should start either with "
                                     "'intent_' or 'entity_' but found: %s"
                                     % stem)

        intents_datasets = [IntentDataset.from_file(f)
                            for f in intent_filepaths]

        entities = [CustomEntity.from_file(f) for f in entity_filepaths]
        entity_names = set(e.name for e in entities)

        # Add entities appearing only in the intents data
        for intent_data in intents_datasets:
            for entity_name in intent_data.entities_names:
                if entity_name not in entity_names:
                    entity_names.add(entity_name)
                    entities.append(create_entity(entity_name))
        return cls(language, intents_datasets, entities)

    @property
    def json(self):
        intents = {intent_data.intent_name: intent_data.json
                   for intent_data in self.intents_datasets}
        entities = {entity.name: entity.json for entity in self.entities}
        return dict(language=self.language, intents=intents, entities=entities)
