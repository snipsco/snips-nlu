import os

from snips_nlu.dataset import validate_dataset

TEST_PATH = os.path.dirname(os.path.abspath(__file__))

EMPTY_DATASET = {"intents": {}, "entities": {}}

SAMPLE_DATASET = {
    "intents": {
        "dummy_intent_1": {
            "utterances": [
                {
                    "data":
                        [
                            {
                                "text": "This is a "
                            },
                            {
                                "text": "dummy_1",
                                "entity": "dummy_entity_1",
                                "slot_name": "dummy_slot_name"
                            },
                            {
                                "text": " query with another "
                            },
                            {
                                "text": "dummy_2",
                                "entity": "dummy_entity_2",
                                "slot_name": "dummy_slot_name2"
                            }
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
                                "entity": "dummy_entity_2",
                                "slot_name": "dummy_slot_name2"
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
                                "text": "This is another "
                            },
                            {
                                "text": "dummy_2_again",
                                "entity": "dummy_entity_2",
                                "slot_name": "dummy_slot_name3"
                            },
                            {
                                "text": "?"
                            }
                        ]
                },
                {
                    "data":
                        [
                            {
                                "text": "dummy_1",
                                "entity": "dummy_entity_1",
                                "slot_name": "dummy_slot_name"
                            }
                        ]
                }
            ]
        },
        "dummy_intent_2": {
            "utterances": [
                {
                    "data":
                        [
                            {
                                "text": "This is a "
                            },
                            {
                                "text": "dummy_3",
                                "entity": "dummy_entity_1",
                                "slot_name": "dummy_slot_name"
                            },
                            {
                                "text": " query from another intent"
                            }
                        ]
                }
            ]
        }
    },
    "entities": {
        "dummy_entity_1": {
            "data": [
                {
                    "value": "dummy_a",
                    "synonyms": ["dummy_a", "dummy 2a", "dummy a",
                                 "2 dummy a"]
                },
                {
                    "value": "dummy_b",
                    "synonyms": ["dummy_b", "dummy_bb", "dummy b"]
                },
                {
                    "value": "dummy\d",
                    "synonyms": ["dummy\d"]
                },
            ],
            "use_synonyms": True,
            "automatically_extensible": False
        },
        "dummy_entity_2": {
            "data": [
                {
                    "value": "dummy_c",
                    "synonyms": ["dummy_c", "dummy_cc", "dummy c",
                                 "3p.m."]
                }
            ],
            "use_synonyms": True,
            "automatically_extensible": False
        }
    }
}

validate_dataset(EMPTY_DATASET)
validate_dataset(SAMPLE_DATASET)
