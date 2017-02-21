import io
import json
import os
import pkgutil
import shutil
import unittest

from collections import defaultdict
from custom_intent_parser.dataset import Dataset
from custom_intent_parser.entity import Entity


try:
    from custom_intent_parser.intent_parser.rasa_intent_parser import (
        RasaIntentParser)
    module_failed = False
except ImportError:
    module_failed = True


def format_rasa_example(example):

    decomposed_query = []
    current_index = 0
    _text = example["text"]
    sorted_entities = sorted(example.get(
        "entities", []), key=lambda entity: entity['start'])
    for entity in sorted_entities:
        if current_index != entity["start"]:
            decomposed_query.append(
                {'text': _text[current_index:entity["start"]]})

        decomposed_query.append({
            'text': _text[entity['start']:entity['end']],
            'entity': entity['entity']
        })
        current_index = entity['end']

    if current_index != len(_text):
        decomposed_query.append({'text': _text[current_index:len(_text)]})

    return {
        "data": decomposed_query
    }


def get_queries():

    with io.open(os.path.join('custom_intent_parser', 'tests',
                              'rasa_nlu_test_data.json'), encoding="utf8") as f:
        data = json.load(f)

    queries = {}

    rasa_examples = data["rasa_nlu_data"]["entity_examples"]

    for example in rasa_examples:

        intent = example["intent"]
        if intent not in queries.keys():
            queries[intent] = []

        queries[intent].append(format_rasa_example(example))

    return queries


def get_entities_from_queries(queries):

    entities_dict = defaultdict(set)

    for intent in queries:
        for query in queries[intent]:
            for span in query['data']:
                if span.get('entity', None):

                    entities_dict[span['entity']].add(span["text"])

    entities_dict = dict((k, list(v)) for k, v in entities_dict.iteritems())

    entities = {k: Entity(k, entries=[{"value": x, "synonyms": [x]} for x in v])
                for (k, v) in entities_dict.iteritems()}

    return entities


class TestFitted(unittest.TestCase):
    _dataset = None

    def setUp(self):

        if module_failed:
            self.skipTest('Rasa intent parser not tested')

        queries = get_queries()
        entities = get_entities_from_queries(queries)

        self._dataset = Dataset(entities=entities, queries=queries)

    def test_not_fitted(self):
        # Given
        parser = RasaIntentParser()

        # Then
        self.assertFalse(parser.fitted())

    def test_fitted(self):
        # Given
        parser = RasaIntentParser()

        # When
        parser = RasaIntentParser().fit(self._dataset)

        # Then
        self.assertTrue(parser.fitted())


class TestRegexIntentParser(unittest.TestCase):

    def setUp(self):

        if module_failed:
            self.skipTest('Rasa intent parser not tested')

        self.load_save_test_dirname = '__rasa_load_save_test'

        if os.path.isdir(self.load_save_test_dirname):
            shutil.rmtree(self.load_save_test_dirname)

        queries = get_queries()
        entities = get_entities_from_queries(queries)

        self._dataset = Dataset(entities=entities, queries=queries)

    def tearDown(self):
        if os.path.isdir(self.load_save_test_dirname):
            shutil.rmtree(self.load_save_test_dirname)

    def test_get_intent(self):
        # Given
        parser = RasaIntentParser().fit(self._dataset)

        # When
        text = "show me chinese restaurants"

        parsed_intent = parser.parse(text)["intent"]
        self.assertTrue(parsed_intent["name"] == 'restaurant_search')
        self.assertTrue(parsed_intent["prob"] > 0)

    def test_load_save(self):
        # Given
        parser1 = RasaIntentParser().fit(self._dataset)


        parser1.save(self.load_save_test_dirname)

        parser2 = RasaIntentParser().load(self.load_save_test_dirname)

        # When
        text = "show me chinese restaurants"

        parsed_intent = parser2.parse(text)["intent"]

        self.assertTrue(parsed_intent["name"] == 'restaurant_search')
        self.assertTrue(parsed_intent["prob"] > 0)


if __name__ == '__main__':
    unittest.main()
