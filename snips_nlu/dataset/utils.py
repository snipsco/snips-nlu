from __future__ import unicode_literals

from future.utils import iteritems, itervalues
from snips_nlu_utils import normalize
from yaml import Loader, SafeLoader

from snips_nlu.constants import (
    DATA, ENTITIES, ENTITY, INTENTS, TEXT, UTTERANCES)
from snips_nlu.entity_parser.builtin_entity_parser import is_gazetteer_entity


def construct_yaml_str(self, node):
    # Override the default string handling function
    # to always return unicode objects
    return self.construct_scalar(node)


Loader.add_constructor("tag:yaml.org,2002:str", construct_yaml_str)
SafeLoader.add_constructor("tag:yaml.org,2002:str", construct_yaml_str)


def extract_utterance_entities(dataset):
    entities_values = {ent_name: set() for ent_name in dataset[ENTITIES]}

    for intent in itervalues(dataset[INTENTS]):
        for utterance in intent[UTTERANCES]:
            for chunk in utterance[DATA]:
                if ENTITY in chunk:
                    entities_values[chunk[ENTITY]].add(chunk[TEXT].strip())
    return {k: list(v) for k, v in iteritems(entities_values)}


def extract_intent_entities(dataset, entity_filter=None):
    intent_entities = {intent: set() for intent in dataset[INTENTS]}
    for intent_name, intent_data in iteritems(dataset[INTENTS]):
        for utterance in intent_data[UTTERANCES]:
            for chunk in utterance[DATA]:
                if ENTITY in chunk:
                    if entity_filter and not entity_filter(chunk[ENTITY]):
                        continue
                    intent_entities[intent_name].add(chunk[ENTITY])
    return intent_entities


def extract_entity_values(dataset, apply_normalization):
    entities_per_intent = {intent: set() for intent in dataset[INTENTS]}
    intent_entities = extract_intent_entities(dataset)
    for intent, entities in iteritems(intent_entities):
        for entity in entities:
            entity_values = set(dataset[ENTITIES][entity][UTTERANCES])
            if apply_normalization:
                entity_values = {normalize(v) for v in entity_values}
            entities_per_intent[intent].update(entity_values)
    return entities_per_intent


def get_text_from_chunks(chunks):
    return "".join(chunk[TEXT] for chunk in chunks)


def get_dataset_gazetteer_entities(dataset, intent=None):
    if intent is not None:
        return extract_intent_entities(dataset, is_gazetteer_entity)[intent]
    return {e for e in dataset[ENTITIES] if is_gazetteer_entity(e)}
