import unittest

from custom_intent_parser.dataset import Dataset
from custom_intent_parser.entity import Entity
from custom_intent_parser.entity_extractor.regex_entity_extractor import (
    RegexEntityExtractor)


def get_queries():
    queries = {
        "dummy_intent_1": [
            {
                "data":
                    [
                        {
                            "text": "This is a "
                        },
                        {
                            "text": "dummy_1",
                            "entity": "dummy_entity_1",
                            "role": "dummy_role"
                        },
                        {
                            "text": " query with another "
                        },
                        {
                            "text": "dummy_2",
                            "entity": "dummy_entity_2"
                        },
                    ]
            },
            {
                "data":
                    [
                        {
                            "text": "This is another "
                        },
                        {
                            "text": "dummy_2_again",
                            "entity": "dummy_entity_2"
                        },
                        {
                            "text": " query."
                        }
                    ]
            },
            {
                "data":
                    [
                        {
                            "text": "dummy_1",
                            "entity": "dummy_entity_1",
                        }
                    ]
            }
        ],
        "dummy_intent_2": [
            {
                "data":
                    [
                        {
                            "text": "This is a "
                        },
                        {
                            "text": "dummy_3",
                            "entity": "dummy_entity_1"
                        },
                        {
                            "text": " query from another intent"
                        }
                    ]
            }
        ]
    }
    return queries


def get_entities():
    entries_1 = [
        {
            "value": "dummy_a",
            "synonyms": ["dummy_a", "dummy_aa", "dummy a", "dummy\s"]
        },
        {
            "value": "dummy_b",
            "synonyms": ["dummy_b", "dummy_bb", "dummy b"]
        },
        {
            "value": "dummy\d",
            "synonyms": ["dummy\d"]
        },
    ]
    entity_1 = Entity("dummy_entity_1", entries=entries_1)

    entries_2 = [
        {
            "value": "dummy_c",
            "synonyms": ["dummy_c", "dummy_cc", "dummy c"]
        }
    ]
    entity_2 = Entity("dummy_entity_2", entries=entries_2)
    return {entity_1.name: entity_1, entity_2.name: entity_2}


class TestRegexEntityExtractor(unittest.TestCase):
    def test_extract_entities(self):
        # Given
        entities = get_entities()
        queries = get_queries()
        dataset = Dataset(entities=entities, queries=queries)
        extractor = RegexEntityExtractor().fit(dataset)

        # When
        expected_entities = {
            "this is a dummy_a query with another dummy_c": [
                {
                    "range": (10, 17),
                    "value": "dummy_a",
                    "entity": "dummy_entity_1",
                    "intent": "dummy_intent_1"
                },
                {
                    "range": (37, 44),
                    "value": "dummy_c",
                    "entity": "dummy_entity_2",
                    "intent": "dummy_intent_1"
                }
            ],
            "this is a whatever query with another whatever": [],
            "this is another dummy_c query.": [
                {
                    "range": (16, 23),
                    "value": "dummy_c",
                    "entity": "dummy_entity_2",
                    "intent": "dummy_intent_1"
                }
            ],
            "dummy\d": [
                {
                    "range": (0, 7),
                    "value": "dummy\d",
                    "entity": "dummy_entity_1",
                    "intent": "dummy_intent_1"
                }
            ],
            "whatever": [],
            "this is a dummy_a query from another intent": [
                {
                    "range": (10, 17),
                    "value": "dummy_a",
                    "entity": "dummy_entity_1",
                    "intent": "dummy_intent_2"
                }
            ]
        }
        entities = dict((t, extractor.get_entities(t))
                        for t in expected_entities.keys())

        # Then
        for t in expected_entities:
            print "Text: %s" % t
            self.assertItemsEqual(entities[t], expected_entities[t])

    def test_extract_entities_with_synonyms(self):
        # Given
        entities = get_entities()
        entities["dummy_entity_1"].use_synonyms = True
        queries = get_queries()
        dataset = Dataset(entities=entities, queries=queries)
        extractor = RegexEntityExtractor().fit(dataset)

        # When
        expected_entities = {
            "this is a dummy a query with another dummy_c": [
                {
                    "range": (10, 17),
                    "value": "dummy a",
                    "entity": "dummy_entity_1",
                    "intent": "dummy_intent_1"
                },
                {
                    "range": (37, 44),
                    "value": "dummy_c",
                    "entity": "dummy_entity_2",
                    "intent": "dummy_intent_1"
                }
            ],
            "this is a whatever query with another whatever": [],
            "this is another dummy_c query.": [
                {
                    "range": (16, 23),
                    "value": "dummy_c",
                    "entity": "dummy_entity_2",
                    "intent": "dummy_intent_1"
                }
            ],
            "dummy_aa": [
                {
                    "range": (0, 8),
                    "value": "dummy_aa",
                    "entity": "dummy_entity_1",
                    "intent": "dummy_intent_1"
                }
            ],
            "whatever": []
        }
        entities = dict((t, extractor.get_entities(t))
                        for t in expected_entities.keys())

        # Then
        for t in expected_entities:
            print "Text: %s" % t
            self.assertItemsEqual(entities[t], expected_entities[t])


if __name__ == '__main__':
    unittest.main()
