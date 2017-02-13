import re
import unittest
from collections import namedtuple

from custom_intent_parser.entity_extractor.regex_entity_extractor import (
    query_to_patterns, RegexEntityExtractor)


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
                            "text": "dummy_2 first",
                            "entity": "dummy_entity_2"
                        },
                        {
                            "text": " with text after it "
                        },
                        {
                            "text": "dummy_1",
                            "entity": "dummy_entity_1",
                            "role": "dummy_role"
                        },
                    ]
            },
            {
                "data":
                    [
                        {
                            "text": "Just dummy_1",
                            "entity": "dummy_entity_1",
                        },
                        {
                            "text": " with another",
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


class TestRegexEntityExtractor(unittest.TestCase):
    def test_query_to_patterns(self):
        # Given
        queries = get_queries()
        # When
        patterns = dict()
        for _, intent_queries in queries.iteritems():
            for query in intent_queries:
                query_patterns = query_to_patterns(query)
                for entity_name, pattern in query_patterns.iteritems():
                    if entity_name not in patterns:
                        patterns[entity_name] = []
                    patterns[entity_name] += pattern

        for entity_name, regexes in patterns.iteritems():
            unique_patterns = set([r.pattern for r in regexes])
            patterns[entity_name] = [re.compile(p, re.IGNORECASE)
                                     for p in unique_patterns]

        # Then
        expected_patterns = {
            "dummy_entity_1": [
                r"This is a (\w+\b)",
                r" with text after it (\w+\b)",
                r"(Just dummy_1)",
                r"( with another)"
            ],
            "dummy_entity_2": [
                r" query with another (\w+\b)",
                r"This is another (\w+\b)",
                r"(\w+\b\s*\w+\b) with text after it "
            ]
        }
        for entity_name, entity_patterns in expected_patterns.iteritems():
            self.assertIn(entity_name, patterns)
            self.assertItemsEqual(entity_patterns,
                                  [r.pattern for r in patterns[entity_name]])

    def test_extract_entities(self):
        # Given
        queries = get_queries()
        mocked_dataset = namedtuple("Dataset", ["queries"])
        dataset = mocked_dataset(queries=queries)
        extractor = RegexEntityExtractor().fit(dataset)

        # When
        expected_entities = {
            "this is a dummy_a entity": [
                {
                    "range": (10, 17),
                    "value": "dummy_a",
                    "entity": "dummy_entity_1"
                }
            ],
            "this is a dummy_a entity with another dummy_b entity and this is"
            " a dummy_c": [
                {
                    "range": (10, 17),
                    "value": "dummy_a",
                    "entity": "dummy_entity_1"
                },
                {
                    "range": (24, 37),
                    "value": " with another",
                    "entity": "dummy_entity_1"
                },
                {
                    "range": (67, 74),
                    "value": "dummy_c",
                    "entity": "dummy_entity_1"
                },
            ],
            "this is an entity with text after it you know?": [
                {
                    "range": (37, 40),
                    "value": "you",
                    "entity": "dummy_entity_1"
                },
                {
                    "range": (8, 17),
                    "value": "an entity",
                    "entity": "dummy_entity_2"
                }
            ],
            "several tokens with text after it ": [
                {
                    "range": (0, 14),
                    "value": "several tokens",
                    "entity": "dummy_entity_2"
                }
            ],
            "if i place just dummy_1 in a sentence it should parser": [
                {
                    "range": (11, 23),
                    "value": "just dummy_1",
                    "entity": "dummy_entity_1"
                }
            ]
        }
        entities = dict((t, extractor.get_entities(t))
                        for t in expected_entities.keys())

        # Then
        for t in expected_entities:
            self.assertItemsEqual(entities[t], expected_entities[t])
