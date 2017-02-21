import argparse
import io
import json
import os
import re
from collections import defaultdict

from dataset import Dataset
from entity import Entity

INTENT_REGEX = re.compile(r"^(?P<intent>[\w]+)\s+")
SLOT_NAME_REGEX = re.compile(r"\{(?P<slot_name>\w+)\}")
SLOT_NAME_SPLIT_REGEX = re.compile(r"\{\w+\}")

ENTITY_ENTRIES_SEP = ";"


def get_entity_chunk(slot_name, intent, ontology):
    if slot_name not in ontology["intents"][intent]["slots"]:
        raise KeyError("Slot '%s' was found in the queries but is missing in "
                       "the intent description" % slot_name)

    entity_name = ontology["intents"][intent]["slots"][slot_name][
        "entityName"]
    if entity_name not in ontology["entities"]:
        raise KeyError("'%s' entityName was found in slot description but is "
                       "missing from the entities description"
                       % entity_name)

    entity_chunk = {
        "text": "dummy_%s" % entity_name,
        "entity": entity_name,
        "role": slot_name
    }
    return entity_chunk


def parse_line(line, ontology):
    intent_match = INTENT_REGEX.match(line)
    if intent_match is None or "intent" not in intent_match.groupdict():
        raise ValueError("No intent in this line: '%s'" % line)
    intent = intent_match.group("intent")
    if intent not in ontology["intents"]:
        raise KeyError("'%s' was found in the queries but is missing in the "
                       "intents description")
    line = INTENT_REGEX.sub("", line)

    entity_matches = SLOT_NAME_REGEX.finditer(line)
    try:
        current_entity_match = next(entity_matches)
    except StopIteration:
        query_data = [{"text": line}]
        return intent, query_data
    num_char = 0
    query_data = []
    for chunk in SLOT_NAME_SPLIT_REGEX.split(line):
        # Loop on adjacent entities
        while current_entity_match is not None and \
                        num_char == current_entity_match.start():
            slot_name = current_entity_match.group("slot_name")
            if slot_name not in ontology["intents"][intent]["slots"]:
                raise KeyError("Found a query with intent '%s' having a "
                               "slot '%s' which is not allowed by the "
                               "intent description." % (intent, slot_name))
            entity_chunk = get_entity_chunk(slot_name, intent, ontology)
            query_data.append(entity_chunk)
            num_char += len("{%s}" % slot_name)
            try:
                current_entity_match = next(entity_matches)
            except StopIteration:
                current_entity_match = None
        if len(chunk) > 0:
            query_data.append({"text": chunk})
            num_char += len(chunk)
    try:
        last_entity_match = next(entity_matches)
        slot_name = last_entity_match.group("slot_name")
        query_data.append(get_entity_chunk(slot_name, intent, ontology))
    except StopIteration:
        pass

    return intent, query_data


def extract_queries(query_utterances_path, ontology):
    with io.open(query_utterances_path, encoding="utf8") as f:
        lines = [l for l in f]
    lines = [l.strip() for l in lines if len(l.strip()) > 0]

    queries = defaultdict(list)
    for l in lines:
        intent, query_data = parse_line(l, ontology)
        queries[intent].append({"data": query_data})
    return queries


def extract_entity_entries(path, use_synonyms):
    with io.open(path, encoding="utf8") as f:
        lines = [l for l in f]
    lines = [l.strip() for l in lines if len(l.strip()) > 0]
    entity_name = os.path.basename(path).replace(".txt", "")
    entries = []
    for l in lines:
        split = l.split(ENTITY_ENTRIES_SEP)
        if not use_synonyms and len(split) != 1:
            raise ValueError("Entity '%s' does not use synonyms but found "
                             "synonyms were found in %s" % (entity_name, path))
        entry = {"value": split[0], "synonyms": split}
        entries.append(entry)
    return entries


def extract_entity(path, ontology):
    entity_name = os.path.basename(path).replace(".txt", "")
    if not os.path.exists(path):
        raise IOError("File not found '%s'" % path)

    entity_description = ontology["entities"][entity_name]

    use_learning = entity_description["automaticallyExtensible"]
    use_synonyms = entity_description["useSynonyms"]
    entries = extract_entity_entries(path, use_synonyms)
    return Entity(entity_name, entries=entries, use_learning=use_learning,
                  use_synonyms=use_synonyms)


def extract_ontology_intents(ontology_data):
    mandatory_intent_keys = ["intent", "slots"]
    mandatory_slot_keys = ["name", "entityName"]
    intents = dict()
    for intent in ontology_data["intents"]:
        parsed_intent = dict()
        for k in mandatory_intent_keys:
            if k not in intent:
                raise KeyError("Missing key '%s' in intent description" % k)
        intent_name = intent["intent"]
        if not isinstance(intent_name, (str, unicode)):
            raise TypeError("Expected 'intent' value to be a str or unicode, "
                            "found %s" % type(intent_name))
        parsed_intent["name"] = intent_name
        parsed_intent["slots"] = dict()
        for slot in intent["slots"]:
            if not isinstance(slot, dict):
                raise TypeError("Expected slot to be an instance of %s, "
                                "but found %s" % type(slot))
            for k in mandatory_slot_keys:
                if k not in slot:
                    raise KeyError("Missing key '%s' in slot description" % k)
            parsed_intent["slots"][slot["name"]] = slot
        intents[parsed_intent["name"]] = parsed_intent

    return intents


def extract_ontology_entities(ontology_data):
    mandatory_keys = ["entityName", "automaticallyExtensible", "useSynonyms"]
    entities = dict()
    for entity in ontology_data["entities"]:
        for k in mandatory_keys:
            if k not in entity:
                raise KeyError("Missing key '%s' in intent description" % k)
        entity_name = entity["entityName"]
        if not isinstance(entity_name, (str, unicode)):
            raise TypeError("Expected 'intent' value to be a str or unicode, "
                            "found %s" % type(entity_name))
        use_learning = entity["automaticallyExtensible"]
        if not isinstance(use_learning, bool):
            raise TypeError("Expected 'automaticallyExtensible' to be a "
                            "boolean but found %s" % type(use_learning))
        use_synonyms = entity["useSynonyms"]
        if not isinstance(use_synonyms, bool):
            raise TypeError("Expected 'useSynonyms' to be a boolean "
                            "but found %s" % type(use_synonyms))
        entities[entity_name] = {
            "entityName": entity_name,
            "automaticallyExtensible": use_learning,
            "useSynonyms": use_synonyms,
        }

    return entities


def extract_ontology(path):
    with io.open(path, encoding="utf8") as f:
        data = json.load(f)
    mandatory_intent_keys = ["intents", "entities"]
    for k in mandatory_intent_keys:
        if k not in data:
            KeyError("Missing '%s' key in ontology" % k)
    ontology = dict()
    if not isinstance(data["intents"], list):
        raise TypeError("Expected ontology's intents to be a list")

    ontology["intents"] = extract_ontology_intents(data)
    ontology["entities"] = extract_ontology_entities(data)
    return ontology


def merge_ontologies(ontologies_dict):
    intents = set()
    slots = set()
    duplicate_intents = set()
    duplicate_slots = set()
    for ontology_file, ontology in ontologies_dict.iteritems():
        ontology_intents = set(ontology["intents"].keys())
        duplicate_intents.update(intents.intersection(ontology_intents))
        intents.update(ontology["intents"].keys())
        ontology_slots = set([s for intent in ontology["intents"]
                              for s in ontology["intents"][intent]["slots"]])
        duplicate_slots.update(slots.intersection(ontology_slots))
        slots.update(ontology_slots)
    if len(duplicate_intents) > 0:
        raise ValueError("Found these intents in several different ontology "
                         "file: %s" % list(duplicate_intents))
    if len(duplicate_slots) > 0:
        raise ValueError("Found these slots in several different ontology "
                         "file: %s" % list(duplicate_slots))
    merged_intents = dict((intent_name, intent)
                          for ontology in ontologies_dict.values()
                          for intent_name, intent
                          in ontology["intents"].iteritems())
    merged_entities = dict((entity_name, entity)
                           for ontology in ontologies_dict.values()
                           for entity_name, entity
                           in ontology["entities"].iteritems())
    return {"intents": merged_intents, "entities": merged_entities}


def dataset_from_asset_directories(assets_dirs, dataset_path):
    if isinstance(assets_dirs, (str, unicode)):
        assets_dirs = [assets_dirs]

    ontologies = dict()
    for assets_dir in assets_dirs:
        json_files = [f for f in os.listdir(assets_dir) if
                      f.endswith(".json")]
        if len(json_files) != 1:
            raise ValueError(
                "Expected 1 json ontology file, found %s in %s"
                % (len(json_files), assets_dir))
        ontology_path = os.path.join(assets_dir, json_files[0])
        ontologies[assets_dir] = extract_ontology(ontology_path)

    # Just to run the mergeability of ontologies
    _ = merge_ontologies(ontologies)

    entities = []
    queries = dict()
    sample_utterance_filename = "SamplesUtterances.txt"
    for assets_dir, ontology in ontologies.iteritems():
        for entity_name in ontology["entities"]:
            entity_utterances_path = os.path.join(assets_dir,
                                                  "%s.txt" % entity_name)
            if not os.path.exists(entity_utterances_path):
                raise IOError("Missing %s file" % entity_utterances_path)
            entities.append(extract_entity(entity_utterances_path, ontology))
        query_utterance_path = os.path.join(assets_dir,
                                            sample_utterance_filename)
        if not os.path.exists(query_utterance_path):
            raise ValueError("%s does not exist" % query_utterance_path)
        queries.update(extract_queries(query_utterance_path, ontology))

    dataset = Dataset(entities, queries)
    dataset.save(dataset_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rCreate a dataset from a"
                                                 " text file")
    parser.add_argument("assets_dirs", metavar='N', type=str, nargs='+',
                        help="List of paths to the assets directories")
    parser.add_argument("dataset_path", help="Output path to the dataset")
    args = parser.parse_args()
    dataset_from_asset_directories(**vars(args))
