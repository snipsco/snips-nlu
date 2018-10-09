from __future__ import unicode_literals

from pathlib import Path

# package
ROOT_PATH = Path(__file__).parent.parent
PACKAGE_NAME = "snips_nlu"
DATA_PACKAGE_NAME = "data"
DATA_PATH = ROOT_PATH / PACKAGE_NAME / DATA_PACKAGE_NAME
PACKAGE_PATH = ROOT_PATH / PACKAGE_NAME

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
ENTITY_KIND = "entity_kind"
RESOLVED_VALUE = "resolved_value"
SLOT_NAME = "slot_name"
TEXT = "text"
UTTERANCES = "utterances"
LANGUAGE = "language"
VALUE = "value"
NGRAM = "ngram"
TOKEN_INDEXES = "token_indexes"
CAPITALIZE = "capitalize"
UNKNOWNWORD = "unknownword"
VALIDATED = "validated"
START = "start"
END = "end"
BUILTIN_ENTITY_PARSER = "builtin_entity_parser"
CUSTOM_ENTITY_PARSER = "custom_entity_parser"
PARSER_THRESHOLD = "parser_threshold"

# resources
STOP_WORDS = "stop_words"
NOISE = "noise"
GAZETTEERS = "gazetteers"
STEMS = "stems"
CUSTOM_ENTITY_PARSER_USAGE = "custom_entity_parser_usage"
WORD_CLUSTERS = "word_clusters"
GAZETTEER_ENTITIES = "gazetteer_entities"
RESOURCES_DIR = "resources_dir"

# builtin entities
SNIPS_AMOUNT_OF_MONEY = "snips/amountOfMoney"
SNIPS_DATETIME = "snips/datetime"
SNIPS_DURATION = "snips/duration"
SNIPS_NUMBER = "snips/number"
SNIPS_ORDINAL = "snips/ordinal"
SNIPS_PERCENTAGE = "snips/percentage"
SNIPS_TEMPERATURE = "snips/temperature"

# languages
LANGUAGE_DE = "de"
LANGUAGE_EN = "en"
LANGUAGE_ES = "es"
LANGUAGE_FR = "fr"
LANGUAGE_IT = "it"
LANGUAGE_JA = "ja"
LANGUAGE_KO = "ko"
