from intent_parser import IntentParser
from snips_nlu.result import ParsedSlot
from snips_nlu.slot_filler.crf_utils import tags_to_labels, OUTSIDE


def tokenize(text):
    return text.split()


def get_slot_name_to_entity(dataset):
    slot_name_to_entity = dict()
    for intent in dataset["intents"].values():
        for utterance in intent["utterances"]:
            for chunk in utterance["data"]:
                if "slot_name" in chunk:
                    slot_name_to_entity[chunk["slot_name"]] = chunk["entity"]
    return slot_name_to_entity


class CRFIntentParser(IntentParser):
    def __init__(self, intent_classifier, crf_taggers):
        super(CRFIntentParser, self).__init__()
        self.intent_classifier = intent_classifier
        self.crf_taggers = crf_taggers
        self.slot_name_to_entity = None

    def get_intent(self, text):
        if not self.fitted:
            raise ValueError("CRFIntentParser must be fitted before "
                             "`get_intent` is called")
        return self.intent_classifier.get_intent(text)

    def get_slots(self, text, intent=None):
        if intent is None:
            raise ValueError("intent can't be None")
        if not self.fitted:
            raise ValueError("CRFIntentParser must be fitted before "
                             "`get_slots` is called")
        if intent not in self.intent_classifier:
            raise KeyError("Invalid intent '%s'" % intent)
        tokens = tokenize(text)
        tagger = self.crf_taggers[intent]

        tags = tagger.get_tags(tokens)
        labels = tags_to_labels(tags, tagging=tagger.tagging)

        slots = []
        current_slot_start_index = 0
        for i in xrange(len(labels)):
            slot_name = labels[i]
            if slot_name != labels[current_slot_start_index]:
                if slot_name != OUTSIDE:
                    rng = (
                        tokens[current_slot_start_index].start,
                        tokens[i - 1].end)
                    slot = ParsedSlot(
                        match_range=rng,
                        value=text[rng[0]: rng[1]],
                        entity=self.slot_name_to_entity[slot_name],
                        slot_name=slot_name
                    )
                    slots.append(slot)
                current_slot_start_index = i

        if current_slot_start_index < len(labels) - 1 \
                and labels[current_slot_start_index] != OUTSIDE:
            rng = (tokens[current_slot_start_index].start, tokens[-1].end)
            slot = ParsedSlot(
                match_range=rng,
                value=text[rng[0]: rng[1]],
                entity=self.slot_name_to_entity[labels[-1]],
                slot_name=labels[-1]
            )
            slots.append(slot)

        return slots

    @property
    def fitted(self):
        return self.intent_classifier.fitted and all(
            slot_filler.fitted for slot_filler in self.crf_taggers)

    def fit(self, dataset):
        self.slot_name_to_entity = get_slot_name_to_entity(dataset)
        self.intent_classifier = self.intent_classifier.fit(dataset)

        for intent_name in dataset["intents"]:
            intent_utterances = dataset["intents"][intent_name]["utterances"]
            crf_samples = [utterance_to_sample(u["data"]) for u in
                           intent_utterances]
            self.crf_taggers[intent_name] = self.crf_taggers[
                intent_name].fit(crf_samples)
        return self


def utterance_to_sample(query):
    tokens, labels = [], []
    for i, chunk in enumerate(query):
        chunk_tokens = tokenize(chunk["text"])
        tokens += chunk_tokens
        if "slot_name" not in chunk:
            labels += [None for _ in xrange(len(chunk_tokens))]
        else:
            slot_name = chunk["slot_name"]
            labels += [slot_name for _ in xrange(len(chunk_tokens))]
    return {"tokens": tokens, "labels": labels}
