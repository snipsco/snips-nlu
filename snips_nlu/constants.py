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
RES_SLOT_NAME = "slot_name"
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
SLOT_NAME = "slot_name"
PROBABILITY = "probability"
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
STEMS = "stems"
WORD_CLUSTERS = "word_clusters"
SUPPORTED_LANGUAGES = "supported_languages"
CAPITALIZE = "capitalize"
VERBS_LEXEMES = "verbs_lexemes"
WORDS_INFLECTION = "words_inflection"
NOISE = "noise"
UNKNOWNWORD = "unknownword"
VALIDATED = "validated"
START = "start"
END = "end"

# resources
STOP_WORDS = "stop_words"
NOISE = "noise"
GAZETTEERS = "gazetteers"
STEMS = "stems"
WORD_CLUSTERS = "word_clusters"
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
LANGUAGE_JA = "ja"
LANGUAGE_KO = "ko"
