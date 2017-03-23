from intent_parser import CustomIntentParser
from snips_nlu.result import ParsedEntity
from snips_nlu.slot_filler.crf_slot_filler import (CRFSlotFiller,
                                                   default_crf_model)
from snips_nlu.slot_filler.crf_utils import (
    BEGINNING_PREFIX, UNIT_PREFIX, LAST_PREFIX, INSIDE_PREFIX, OUTSIDE)
from snips_nlu.slot_filler.feature_functions import default_features


def check_fitted(parser):
    if parser.intent_classifier is None or parser.crf_slot_fillers is None:
        raise ValueError("Intent parser is not fitted")


def tokenize(text):
    return text.split()


def tag_name_to_slot_name(tag):
    return tag[2:]


def end_of_boi_slot(tags, i):
    if i + 1 == len(tags):
        if tags[i] != OUTSIDE:
            return True
        else:
            return False
    else:
        if tags[i] == OUTSIDE:
            return False
        else:
            if tags[i + 1].startswith(INSIDE_PREFIX):
                return False
            else:
                return True


def tags_to_slots_with_bio(tokens, tags):
    slots = []
    current_slot_start = 0
    for i, tag in enumerate(tags):
        if tag.startswith(BEGINNING_PREFIX):
            current_slot_start = i
        if end_of_boi_slot(tags, i):
            slots.append({
                "range": (current_slot_start, i + 1),
                "value": " ".join(tokens[current_slot_start: i + 1]),
                "slot_name": tag_name_to_slot_name(tag)
            })
    return slots


def tags_to_slots_with_bilou(tokens, tags):
    slots = []
    current_slot_start = 0
    for i, tag in enumerate(tags):
        if tag.startswith(UNIT_PREFIX):
            slots.append({"range": (i, i + 1), "value": tokens[i],
                          "slot_name": tag_name_to_slot_name(tag)})
        if tag.startswith(BEGINNING_PREFIX):
            current_slot_start = i
        if tag.startswith(LAST_PREFIX) :
            slots.append({"range": (current_slot_start, i + 1),
                          "value": " ".join(tokens[current_slot_start: i + 1]),
                          "slot_name": tag_name_to_slot_name(tag)})
    return slots


def tags_to_slots(tokens, tags, use_bilou):
    return tags_to_slots_with_bilou(tokens, tags) \
        if use_bilou else tags_to_slots_with_bio(tokens, tags)


def get_slot_name_to_entity(dataset):
    slot_name_to_entity = dict()
    for intent in dataset["intents"].values():
        for utterance in intent["utterances"]:
            for chunk in utterance["data"]:
                if "slot_name" in chunk:
                    slot_name_to_entity[chunk["slot_name"]] = chunk["entity"]
    return slot_name_to_entity


def utterance_to_bilou_sample(query):
    tokens, labels = [], []
    for i, chunk in enumerate(query):
        chunk_tokens = tokenize(chunk["text"])
        tokens += chunk_tokens
        if "slot_name" not in chunk:
            labels += [OUTSIDE for _ in xrange(len(chunk_tokens))]
        else:
            slot_name = chunk["slot_name"]
            if len(chunk_tokens) == 1:
                labels.append(UNIT_PREFIX + slot_name)
            else:
                labels.append(BEGINNING_PREFIX + slot_name)
                labels += [INSIDE_PREFIX + slot_name for _
                           in xrange(1, len(chunk_tokens) - 1)]
                labels.append(LAST_PREFIX + slot_name)
    return {"tokens": tokens, "labels": labels}


def utterance_to_bio_sample(query):
    tokens, labels = [], []
    for i, chunk in enumerate(query):
        chunk_tokens = tokenize(chunk["text"])
        tokens += chunk_tokens
        if "slot_name" not in chunk:
            labels += [OUTSIDE for _ in xrange(len(chunk_tokens))]
        else:
            slot_name = chunk["slot_name"]
            labels.append(BEGINNING_PREFIX + slot_name)
            labels += [INSIDE_PREFIX + slot_name for _
                       in xrange(1, len(chunk_tokens))]
    return {"tokens": tokens, "labels": labels}


def utterance_to_sample(query, use_bilou):
    return utterance_to_bilou_sample(query) if use_bilou \
        else utterance_to_bio_sample(query)


class CRFIntentParser(CustomIntentParser):
    def __init__(self, intent_classifier=None, crf_slot_fillers=None,
                 use_bilou=True):
        super(CRFIntentParser, self).__init__()
        self.intent_classifier = intent_classifier
        self.crf_slot_fillers = crf_slot_fillers
        self.slot_name_to_entity = None
        self.use_bilou = use_bilou

    def get_intent(self, text):
        check_fitted(self)
        return self.intent_classifier.get_intent(text)

    def get_entities(self, text, intent=None):
        if intent is None:
            raise ValueError("intent can't be None")
        check_fitted(self)
        if intent not in self.intent_classifier:
            raise KeyError("Invalid intent '%s'" % intent)
        tokens = tokenize(text)
        slot_filler = self.crf_slot_fillers[intent]

        tags = slot_filler.get_slots(tokens)
        slots = tags_to_slots(tokens, tags, self.use_bilou)

        return [ParsedEntity(s["range"], s["value"],
                             self.slot_name_to_entity[s["slot_name"]],
                             s["slot_name"]) for s in slots]

    @property
    def fitted(self):
        return self.intent_classifier is not None \
               and self.crf_slot_fillers is not None

    def fit(self, dataset, crf_model=default_crf_model(), features=None):
        if features is None:
            features = default_features(self.use_bilou)
        self.slot_name_to_entity = get_slot_name_to_entity(dataset)
        self.intent_classifier.fit(dataset)

        self.crf_slot_fillers = dict()
        for intent_name in dataset["intents"]:
            intent_utterances = dataset["intents"][intent_name]["utterances"]
            crf_samples = [utterance_to_sample(u["data"], self.use_bilou)
                           for u in intent_utterances]
            crf_slot_filler = CRFSlotFiller(crf_model, features,
                                            self.use_bilou)
            self.crf_slot_fillers[intent_name] = crf_slot_filler.fit(
                crf_samples)
        return self
