import argparse
import json

from nlu_dataset.builtin_entities import BUILTIN_ENTITIES
from nlu_dataset.intent_dataset import IntentDataset


class AssistantDataset(object):
    """Dataset of an assistant

    Merges a list of :class:AssistantDataset into a single dataset ready to be
    used by Snips NLU

    Attributes:
        :class:AssistantDataset.language: language of the dataset
        :class:AssistantDataset.intent_datasets: list of :class:IntentDataset
        :class:AssistantDataset.json: The dataset in json format
    """

    def __init__(self, language, intent_datasets):
        self.language = language
        self.intent_datasets = intent_datasets

    @classmethod
    def from_files(cls, language, file_names):
        datasets = [IntentDataset.from_file(language, f) for f in file_names]
        return cls(language, datasets)

    @property
    def json(self):
        intent_datasets_json = {d.intent_name: d.json
                                for d in self.intent_datasets}

        intents = {
            intent_name: {"utterances": dataset_json["utterances"]}
            for intent_name, dataset_json in intent_datasets_json.items()
        }
        entities = dict()

        for dataset in intent_datasets_json.values():
            for entity, data in dataset["entities"].items():
                if entity not in entities:
                    entities[entity] = data
                elif entity not in BUILTIN_ENTITIES:
                    entities["data"].append(data["data"])

        return dict(language=self.language,
                    intents=intents,
                    entities=entities)


def main_generate_dataset():
    parser = argparse.ArgumentParser(
        description="Create a Snips dataset from text friendly files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--language", metavar="", type=str, default="en",
                        help="Language of the assistant")
    parser.add_argument("intent_files", type=str, nargs="+",
                        help="List of intent files containing utterances")
    args = parser.parse_args()
    dataset = AssistantDataset.from_files(args.language, args.intent_files)
    print(json.dumps(dataset.json, indent=2))
