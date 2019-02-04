from __future__ import unicode_literals

import json

from snips_nlu.common.utils import json_string
from snips_nlu.preprocessing import tokenize
from snips_nlu.result import unresolved_slot
from snips_nlu.slot_filler import SlotFiller


@SlotFiller.register("keyword_slot_filler")
class KeywordSlotFiller(SlotFiller):
    def __init__(self, config=None, **shared):
        super(KeywordSlotFiller, self).__init__(config, **shared)
        self.slots_keywords = None
        self.language = None

    @property
    def fitted(self):
        return self.slots_keywords is not None

    def fit(self, dataset, intent):
        self.language = dataset["language"]
        self.slots_keywords = dict()
        utterances = dataset["intents"][intent]["utterances"]
        for utterance in utterances:
            for chunk in utterance["data"]:
                if "slot_name" in chunk:
                    text = chunk["text"]
                    if self.config.get("lowercase", False):
                        text = text.lower()
                    self.slots_keywords[text] = [
                        chunk["entity"],
                        chunk["slot_name"]
                    ]
        return self

    def get_slots(self, text):
        tokens = tokenize(text, self.language)
        slots = []
        for token in tokens:
            normalized_value = token.value
            if self.config.get("lowercase", False):
                normalized_value = normalized_value.lower()
            if normalized_value in self.slots_keywords:
                entity = self.slots_keywords[normalized_value][0]
                slot_name = self.slots_keywords[normalized_value][1]
                slot = unresolved_slot((token.start, token.end), token.value,
                                       entity, slot_name)
                slots.append(slot)
        return slots

    def persist(self, path):
        model = {
            "language": self.language,
            "slots_keywords": self.slots_keywords,
            "config": self.config.to_dict()
        }
        with path.open(mode="w") as f:
            f.write(json_string(model))

    @classmethod
    def from_path(cls, path, **shared):
        with path.open() as f:
            model = json.load(f)
        slot_filler = cls()
        slot_filler.language = model["language"]
        slot_filler.slots_keywords = model["slots_keywords"]
        slot_filler.config = cls.config_type.from_dict(model["config"])
        return slot_filler
