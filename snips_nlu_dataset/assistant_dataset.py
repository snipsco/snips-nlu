# coding=utf-8
from __future__ import unicode_literals, print_function

import argparse
import json

from snips_nlu_dataset.entities import CustomEntity, create_entity
from snips_nlu_dataset.intent_dataset import IntentDataset


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
    def from_files(cls, language, intents_file_names=None,
                   entities_file_names=None):
        """Creates an :class:`.AssistantDataset` from a language and a list of
        intent and entity files

        Args:
            language (str): language of the assistant
            intents_file_names (list of str, optional): names of intent files.
                The assistant will associate each file to an intent, the name
                of the file being the intent name.
            entities_file_names (list of str, optional): names of custom entity
                files. The assistant will associate each file to an entity, the
                name of the file being the entity name.
        """
        if intents_file_names is None:
            intents_file_names = []
        intents_datasets = [IntentDataset.from_file(f)
                            for f in intents_file_names]

        if entities_file_names is None:
            entities_file_names = []
        entities = [CustomEntity.from_file(f) for f in entities_file_names]
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


def main_generate_dataset():
    parser = argparse.ArgumentParser(
        description="Create a Snips dataset from text friendly files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--language", metavar="", type=str, default="en",
                        help="Language of the assistant")
    parser.add_argument("--intent-files", type=str, nargs="+",
                        help="List of intent files containing utterances")
    parser.add_argument("--entity-files", type=str, nargs="+",
                        help="List of entity files")
    args = parser.parse_args()
    dataset = AssistantDataset.from_files(args.language, args.intent_files,
                                          args.entity_files)
    print(json.dumps(dataset.json, indent=2, sort_keys=True))


if __name__ == '__main__':
    main_generate_dataset()
