import os

from intent_parser import IntentParser
from ..utils import LimitedSizeDict
from ..result import parsed_entity

from snips.preprocessing import tokenize
from snips.attribute_extraction import AttributeExtraction
from snips.intent_classifier import IntentClassifier


def parse_entity(text, raw_entities):
    entities = []
    for intent in raw_entities:
        tokens = sorted(raw_entities[intent],
                        key=lambda token: token['startIndex'])
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

        entities.extend([parsed_entity((start_index, end_index),
            text[start_index:end_index], entity=intent, slot_name=intent)
            for (start_index, end_index) in spans])

    return entities


class BuiltinIntentParser(IntentParser):
    def __init__(self, resources_path, intents,
                 cache=None, cache_size=100):
        self.configs_path = os.path.join(resources_path, 'configurations')
        self.gazetteers_path = os.path.join(resources_path, 'gazetteers')
        self._intents = None
        self.intents = intents

        if cache is None:
            cache = LimitedSizeDict(size_limit=cache_size)
        self._cache = cache

    def is_valid_intent(self, intent):
        return os.path.exists(os.path.join(self.configs_path, '%s.pb' % intent))

    @property
    def intents(self):
        return self._intents

    @intents.setter
    def intents(self, value):
        for intent in value:
            if not self.is_valid_intent(intent):
                raise IOError('The built-in intent `%s` not found in the '
                    'resource folder `%s`.' % (intent, self.configs_path))

        self._intents = value

    def parse(self, text):
        if text not in self._cache:
            self._cache[text] = self._parse(text)
        return self._cache[text]

    def _parse(self, text):
        if not self.intents:
            return {'text': text, 'intent': None, 'entities': []}

        intent = self.get_intent(text)
        entities = self.get_entities(text, intent=intent.get('name'))

        return {'text': text, 'intent': intent, 'entities': entities}

    def get_intent(self, text):
        if not self.intents:
            return None

        tokenized_text = tokenize({'text': unicode(text)})
        max_proba, best_intent = -1., None
        for intent in self.intents:
            intent_classifier = IntentClassifier(
                intent_config_file=os.path.join(
                    self.configs_path, '%s.pb' % intent),
                gazetteers_dir=self.gazetteers_path
            )
            proba = intent_classifier.transform(tokenized_text)

            if (max_proba < 0.) or (proba > max_proba):
                max_proba = proba
                best_intent = intent

        return {'name': best_intent, 'proba': max_proba}

    def get_entities(self, text, intent=None):
        if intent is None:
            intent = self.get_intent(text).get('name')

        if not self.is_valid_intent(intent):
            raise IOError('The built-in intent `%s` not found in the '
                    'resource folder `%s`.' % (intent, self.configs_path))

        tokenized_text = tokenize({'text': unicode(text)})
        entity_extractor = AttributeExtraction(
            intent_config_file=os.path.join(
                self.configs_path, '%s.pb' % intent),
            gazetteers_dir=self.gazetteers_path
        )
        entities = entity_extractor.transform(tokenized_text)

        return parse_entity(unicode(text), entities)
