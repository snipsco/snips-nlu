# coding=utf-8
from __future__ import unicode_literals, print_function

import argparse
import json
import os
from copy import deepcopy

from future.utils import iteritems

from snips_nlu_dataset.custom_entities import CustomEntity
from snips_nlu_dataset.intent_dataset import IntentDataset
from snips_nlu.builtin_entities import is_builtin_entity


class AssistantDataset(object):
    """Dataset of an assistant

    Merges a list of :class:AssistantDataset into a single dataset ready to be
    used by Snips NLU

    Attributes:
        :class:AssistantDataset.language: language of the dataset
        :class:AssistantDataset.intent_datasets: list of :class:IntentDataset
        :class:AssistantDataset.entities: dict of :class:CustomEntity
        :class:AssistantDataset.json: The dataset in json format
    """

    def __init__(self, language, intent_datasets, entities):
        self.language = language
        self.intent_datasets = intent_datasets
        self.entities = entities

    @classmethod
    def from_files(cls, language, intents_file_names=None,
                   entities_file_names=None):
        """Creates an :class:AssistantDataset from a language and a list of
        text files

        The assistant will associate each file to an intent, the name of the
        file being the intent name.
        """
        if intents_file_names is None:
            intents_file_names = []
        datasets = [IntentDataset.from_file(language, f) for f in
                    intents_file_names]
        if entities_file_names is None:
            entities_file_names = []
        entities = {
            os.path.splitext(os.path.basename(f))[0]: CustomEntity.from_file(f)
            for f in entities_file_names
        }
        return cls(language, datasets, entities)

    @property
    def json(self):
        intent_datasets_json = {d.intent_name: d.json
                                for d in self.intent_datasets}
        intents = {
            intent_name: {
                "utterances": dataset_json["utterances"]
            }
            for intent_name, dataset_json in iteritems(intent_datasets_json)
        }
        ents = deepcopy(self.entities)
        ents_values = dict()
        for entity_name, entity in iteritems(self.entities):
            ents_values[entity_name] = set(a.value for a in entity.utterances)
            if entity.use_synonyms:
                ents_values[entity_name].update(
                    set(t for s in entity.utterances for t in s.synonyms))

        for dataset in self.intent_datasets:
            for ent_name, ent in iteritems(dataset.entities):
                if ent_name not in ents:
                    ents[ent_name] = ent
                elif not is_builtin_entity(ent_name):
                    for u in ent.utterances:
                        if u.value not in ents_values:
                            ents[ent_name].utterances.append(u)
        ents = {
            entity_name: entity.json
            for entity_name, entity in iteritems(ents)
        }
        return dict(language=self.language, intents=intents, entities=ents)


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
    print(json.dumps(dataset.json, indent=2))


if __name__ == '__main__':
    main_generate_dataset()
