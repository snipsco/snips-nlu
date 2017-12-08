from __future__ import unicode_literals

import os

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESOURCE_PACKAGE_NAME = "snips-nlu-resources"
PACKAGE_NAME = "snips_nlu"
RESOURCES_PATH = os.path.join(ROOT_PATH, PACKAGE_NAME, RESOURCE_PACKAGE_NAME)
PACKAGE_PATH = os.path.join(ROOT_PATH, PACKAGE_NAME)

INTENT_NAME = "intent_name"
PROBABILITY = "probability"
PARSED_INTENT = "intent"
PARSED_SLOTS = "slots"
TEXT = "text"
NORMALIZED_TEXT = "normalized_text"
AUTOMATICALLY_EXTENSIBLE = "automatically_extensible"
USE_SYNONYMS = "use_synonyms"
SYNONYMS = "synonyms"
DATA = "data"
INTENTS = "intents"
ENTITIES = "entities"
ENTITY = "entity"
SLOT_NAME = "slot_name"
UTTERANCES = "utterances"
LANGUAGE = "language"
MATCH_RANGE = "range"
VALUE = "value"
BUILTIN_PARSER = "builtin_parser"
BUILTIN_PATH = "builtin_path"
BUILTIN_BINARY = "builtin_binary"
LABEL = "label"
RUSTLING_DIM_KIND = "rustling_dim_kind"
NGRAM = "ngram"
TOKEN_INDEXES = "token_indexes"
GAZETTEERS = "gazetteers"
STOP_WORDS = "stop_words"
WORD_CLUSTERS = "word_clusters"
SNIPS_NLU_VERSION = "snips_nlu_version"
SUPPORTED_LANGUAGES = "supported_languages"
CAPITALIZE = "capitalize"
NOISE = "noise"
UNKNOWNWORD = "unknownword"
