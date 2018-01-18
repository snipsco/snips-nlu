from __future__ import unicode_literals

import os

# package
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESOURCE_PACKAGE_NAME = "snips-nlu-resources"
PACKAGE_NAME = "snips_nlu"
RESOURCES_PATH = os.path.join(ROOT_PATH, PACKAGE_NAME, RESOURCE_PACKAGE_NAME)
PACKAGE_PATH = os.path.join(ROOT_PATH, PACKAGE_NAME)

# result
RES_INPUT = "input"
RES_INTENT = "intent"
RES_SLOTS = "slots"
RES_INTENT_NAME = "intentName"
RES_PROBABILITY = "probability"
RES_SLOT_NAME = "slotName"
RES_ENTITY = "entity"
RES_VALUE = "value"
RES_RAW_VALUE = "rawValue"
RES_MATCH_RANGE = "range"

# miscellaneous
AUTOMATICALLY_EXTENSIBLE = "automatically_extensible"
USE_SYNONYMS = "use_synonyms"
SYNONYMS = "synonyms"
DATA = "data"
INTENTS = "intents"
ENTITIES = "entities"
ENTITY = "entity"
SLOT_NAME = "slot_name"
TEXT = "text"
UTTERANCES = "utterances"
LANGUAGE = "language"
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
