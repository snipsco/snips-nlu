import os

from snips.attribute_extraction import AttributeExtraction
from snips.intent_classifier import IntentClassifier
from snips.preprocessing import tokenize

from intent_parser import IntentParser, BUILTIN_PARSER_TYPE
from ..result import ParsedEntity, Result, IntentClassificationResult


def parse_entity(text, raw_entities):
    entities = []
    for entity in raw_entities:
        tokens = sorted(raw_entities[entity],
                        key=lambda _token: _token['startIndex'])
        start_index, end_index = -1, -1
        spans = []
        for token in tokens:
            if (end_index < 0) or ((token['startIndex'] - end_index) > 1):
                if end_index > 0:
                    spans.append((start_index, end_index))
                start_index = token['startIndex']
            end_index = token['endIndex']
        if end_index > 0:
            spans.append((start_index, end_index))

        entities.extend([ParsedEntity(match_range=(start_index, end_index),
                                      value=text[start_index:end_index],
                                      entity=entity,
                                      slot_name=entity)
                         for (start_index, end_index) in spans])

    return entities


class BuiltinIntentParser(IntentParser):
    def __init__(self, config_path, resources_dir):
        intent_name = os.path.splitext(os.path.basename(config_path))[0]
        super(BuiltinIntentParser, self).__init__(intent_name,
                                                  BUILTIN_PARSER_TYPE)
        self.config_path = config_path
        self.resources_dir = resources_dir

    def parse(self, text):
        intent = self.get_intent(text)
        entities = self.get_entities(text, intent=intent['name'])
        return Result(text=text, parsed_intent=intent, parsed_entities=entities)

    def get_intent(self, text):
        tokenized_text = tokenize({'text': unicode(text)})
        intent_classifier = IntentClassifier(
            intent_config_file=self.config_path,
            gazetteers_dir=self.resources_dir
        )
        proba = intent_classifier.transform(tokenized_text)
        return IntentClassificationResult(intent_name=self.intent_name,
                                          probability=proba)

    def get_entities(self, text, intent=None):
        tokenized_text = tokenize({'text': unicode(text)})
        entity_extractor = AttributeExtraction(
            intent_config_file=self.config_path,
            gazetteers_dir=self.resources_dir
        )
        entities = entity_extractor.transform(tokenized_text)
        return parse_entity(unicode(text), entities)
