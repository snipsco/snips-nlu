import io
import json
import os
import pkgutil
import shutil
import unittest

from custom_intent_parser.dataset import Dataset
from custom_intent_parser.entity import Entity

try:
    from custom_intent_parser.intent_parser.rasa_intent_parser import RasaIntentParser
    module_failed = False
except ImportError:
    module_failed = True


def format_rasa_example(example):

    decomposed_query = []
    current_index = 0
    _text=example["text"]
    sorted_entities=sorted(example.get("entities",[]), key=lambda entity: entity['start']) 
    for entity in sorted_entities:
        if current_index != entity["start"]:
            decomposed_query.append({'text': _text[current_index:entity["start"]]})

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

    with open('custom_intent_parser/tests/rasa_nlu_test_data.json') as data_file:    
        data = json.load(data_file)

    queries={}
    entities=[]

    rasa_examples=data["rasa_nlu_data"]["entity_examples"]

    for example in rasa_examples:

        intent=example["intent"]
        if intent not in queries.keys():
            queries[intent]=[]

        queries[intent].append(format_rasa_example(example))

    return queries


def get_entities_from_queries(queries):

    entities_dict={}

    for intent in queries:
        for query in queries[intent]:
            for span in query['data']:
                if span.get('entity', None):

                    entity_name=span['entity']
                    if entity_name not in entities_dict:
                        entities_dict[entity_name]=[]

                    if span['text'] not in entities_dict[entity_name]:
                        entities_dict[entity_name].append(span['text'])

    entities={}
    for entity_name in entities_dict.keys():
        entities[entity_name]=Entity(entity_name, entries=[
            {
                "value": value,
                "synonyms": [value]
            }
        for value in entities_dict[entity_name]])

    return entities



class TestRegexIntentParser(unittest.TestCase):
    _dataset = None

    def setUp(self):

        if module_failed:
            self.skipTest('Rasa intent parser not tested')

        queries = get_queries()

        # so far we don't use other intents than the one of the intent we're training
        # a model for
        queries = {'restaurant_search': queries['restaurant_search']}

        entities = get_entities_from_queries(queries)

        self._dataset = Dataset(entities=entities, queries=queries)


    def test_get_intent(self):
        # Given
        parser = RasaIntentParser().fit(self._dataset)

        # When
        text="show me chinese restaurants"

        parsed_intent=parser.parse(text)["intent"]
        self.assertTrue(parsed_intent["intent"]=='restaurant_search' and parsed_intent["proba"]>0)


    def test_load_save(self):
        # Given
        parser1 = RasaIntentParser().fit(self._dataset)

        dirname = '__rasa_load_save_test'

        if os.path.isdir(dirname):
            shutil.rmtree(dirname)
        parser1.save(dirname)

        parser2 = RasaIntentParser().load(dirname)
        shutil.rmtree(dirname)

        # When
        text="show me chinese restaurants"

        parsed_intent=parser2.parse(text)["intent"]

        self.assertTrue(parsed_intent["intent"]=='restaurant_search' and parsed_intent["proba"]>0)        


if __name__ == '__main__':
    unittest.main()
