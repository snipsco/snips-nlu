import argparse
import io
import re
from collections import defaultdict

from dataset import Dataset
from entity import Entity

INTENT_REGEX = re.compile(r"^(?P<intent>[\w]+)\s+")
ENTITY_ROLE_REGEX = re.compile(r"@(?P<entity>[\w]+):?(?P<role>[\w]+)?")
ENTITY_ROLE_SPLIT_REGEX = re.compile(r"@[\w]+[:[\w]+]?")


def get_entity_chunk(current_entity_match):
    entity = current_entity_match.group("entity")
    role = current_entity_match.group("role")
    entity_chunk = {"text": "put_a_%s_here" % entity, "entity": entity}
    if role is not None:
        entity_chunk["role"] = role
    return entity_chunk


def parse_line(line):
    intent_match = INTENT_REGEX.match(line)
    if intent_match is None or "intent" not in intent_match.groupdict():
        raise ValueError("No intent in this line: %s" % line)
    intent = intent_match.group("intent")
    line = INTENT_REGEX.sub("", line)

    entity_matches = ENTITY_ROLE_REGEX.finditer(line)
    try:
        current_entity_match = next(entity_matches)
    except StopIteration:
        query_data = [{"text": line}]
        return intent, query_data
    num_char = 0
    query_data = []
    for chunk in ENTITY_ROLE_SPLIT_REGEX.split(line):
        while current_entity_match is not None and \
                        num_char == current_entity_match.start():
            entity_chunk = get_entity_chunk(current_entity_match)
            query_data.append(entity_chunk)
            num_char += len("@%s" % entity_chunk["entity"])
            if "role" in entity_chunk:
                num_char += len(":%s" % entity_chunk["role"])
            try:
                current_entity_match = next(entity_matches)
            except StopIteration:
                current_entity_match = None
        if len(chunk) > 0:
            query_data.append({"text": chunk})
            num_char += len(chunk)
    try:
        last_entity_match = next(entity_matches)
        query_data.append(get_entity_chunk(last_entity_match))
    except StopIteration:
        pass

    return intent, query_data


def dataset_from_text_file(path, dataset_path):
    with io.open(path, encoding="utf8") as f:
        lines = [l for l in f]
    lines = [l.strip() for l in lines if len(l.strip()) > 0]

    queries = defaultdict(list)
    entities = defaultdict(list)
    for l in lines:
        intent, query_data = parse_line(l)
        queries[intent].append({"data": query_data})
        for c in query_data:
            if "entity" not in c:
                continue
            entry = c["text"]
            if all(e != entry for e in entities[c["entity"]]):
                entities[c["entity"]].append({"value": c["text"],
                                              "synonyms": [c["text"]]})
    for entity_name, entry_list in entities.iteritems():
        existing_entries = set()
        unique_entries = list()
        for entry in entry_list:
            if entry["value"] not in existing_entries:
                existing_entries.add(entry["value"])
                unique_entries.append(entry)
        entities[entity_name] = unique_entries

    entities = [Entity(ent_name, entries=entries)
                for ent_name, entries in entities.iteritems()]
    dataset = Dataset(entities, queries)
    dataset.save(dataset_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rCreate a dataset from a"
                                                 " text file")
    parser.add_argument("path", help="Path to the text file")
    parser.add_argument("dataset_path", help="Output path to the dataset")
    args = parser.parse_args()
    dataset_from_text_file(**vars(args))
