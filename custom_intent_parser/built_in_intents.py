import json
import os
import subprocess

from enum import Enum

from utils import ROOT_PATH

BINARY_PATH = os.path.join(ROOT_PATH, "snips-queries-rust", "queries-cli",
                           "target", "debug", "queries-cli")

DATA_PATH = os.path.join(ROOT_PATH, "snips-queries-rust", "data")


class BuiltInIntent(Enum):
    BookRestaurant = {
        "name": "BookRestaurant",
        "slots": {"restaurant", "reservationDatetime", "partySize"}
    }


def get_built_in_intents(text, candidate_intents):
    intent = " ".join([c.value["name"] for c in candidate_intents])
    output = subprocess.check_output(
        [BINARY_PATH, text, "--root_dir", DATA_PATH, "--intent", intent,
         "intents"])
    output = json.loads(output)
    results = list()
    for i, res in enumerate(output):
        intent_name = res["name"]
        try:
            parsed_intent = BuiltInIntent[intent_name]
        except KeyError:
            raise KeyError("Built in intent parser returned unknown intent "
                           "'%s'" % intent_name)
        results.append({"intent": parsed_intent, "prob": res["probability"]})
    return results


def get_built_in_intent_entities(text, intent):
    output = subprocess.check_output(
        [BINARY_PATH, text, "--root_dir", DATA_PATH, "tokens", "--intent_name",
         intent.value["name"]])
    output = json.loads(output)
    results = dict()
    for slot_name, slot_value in output.iteritems():
        if slot_name not in BuiltInIntent[intent.value["name"]].value["slots"]:
            raise KeyError("Unknown slot '%s' for intent '%s'"
                           % (slot_name, intent))
        if len(slot_value) > 0:
            results[slot_name] = slot_value
    return results
