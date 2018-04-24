from __future__ import unicode_literals

from snips_nlu_utils import compute_all_ngrams

from snips_nlu.builtin_entities import is_builtin_entity
from snips_nlu.constants import (
    RES_MATCH_RANGE, INTENTS, UTTERANCES, DATA, ENTITY, ENTITIES, END, START)
from snips_nlu.utils import LimitedSizeDict

_NGRAMS_CACHE = LimitedSizeDict(size_limit=1000)


def get_all_ngrams(tokens):
    key = "<||>".join(tokens)
    if key not in _NGRAMS_CACHE:
        ngrams = compute_all_ngrams(tokens, len(tokens))
        _NGRAMS_CACHE[key] = ngrams
    return _NGRAMS_CACHE[key]


def get_word_chunk(word, chunk_size, chunk_start, reverse=False):
    if chunk_size < 1:
        raise ValueError("chunk size should be >= 1")
    if chunk_size > len(word):
        return None
    start = chunk_start - chunk_size if reverse else chunk_start
    end = chunk_start if reverse else chunk_start + chunk_size
    return word[start:end]


def initial_string_from_tokens(tokens):
    current_index = 0
    s = ""
    for t in tokens:
        if t.start > current_index:
            s += " " * (t.start - current_index)
        s += t.value
        current_index = t.end
    return s


def entity_filter(entity, start, end):
    entity_start = entity[RES_MATCH_RANGE][START]
    entity_end = entity[RES_MATCH_RANGE][END]
    return (entity_start <= start < entity_end) \
           and (entity_start < end <= entity_end)


def get_intent_custom_entities(dataset, intent):
    intent_entities = set()
    for utterance in dataset[INTENTS][intent][UTTERANCES]:
        for c in utterance[DATA]:
            if ENTITY in c:
                intent_entities.add(c[ENTITY])
    custom_entities = dict()
    for ent in intent_entities:
        if not is_builtin_entity(ent):
            custom_entities[ent] = dataset[ENTITIES][ent]
    return custom_entities
