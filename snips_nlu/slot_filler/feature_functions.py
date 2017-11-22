from __future__ import unicode_literals

from snips_nlu.builtin_entities import get_builtin_entities, BuiltInEntity
from snips_nlu.constants import (MATCH_RANGE, TOKEN_INDEXES, NGRAM)
from snips_nlu.languages import Language
from snips_nlu.preprocessing import stem
from snips_nlu.resources import get_word_clusters, get_gazetteer
from snips_nlu.slot_filler.crf_utils import get_scheme_prefix, TaggingScheme
from snips_nlu.slot_filler.de.specific_features_functions import \
    language_specific_features as de_features
from snips_nlu.slot_filler.default.default_features_functions import \
    default_features
from snips_nlu.slot_filler.en.specific_features_functions import \
    language_specific_features as en_features
from snips_nlu.slot_filler.features_utils import get_all_ngrams, get_shape, \
    get_word_chunk, initial_string_from_tokens
from snips_nlu.slot_filler.fr.specific_features_functions import \
    language_specific_features as fr_features
from snips_nlu.slot_filler.ko.specific_features_functions import \
    language_specific_features as ko_features

TOKEN_NAME = "token"


class Feature(object):
    def __init__(self, name, func, feature_type=None, offset=0):
        if name == TOKEN_NAME:
            raise ValueError("'%s' name is reserved" % TOKEN_NAME)
        self._name = name
        self._feature_type = feature_type
        self.function = func
        self.offset = offset

    def get_offset_feature(self, offset):
        if offset == 0:
            return self
        else:
            return Feature(self._name, self.function, self.feature_type,
                           offset)

    @property
    def name(self):
        if self.offset > 0:
            return "%s[+%s]" % (self._name, self.offset)
        elif self.offset < 0:
            return "%s[%s]" % (self._name, self.offset)
        else:
            return self._name

    @property
    def feature_type(self):
        if self._feature_type is not None:
            return self._feature_type
        else:
            return self._name

    def compute(self, token_index, cache):
        if not 0 <= (token_index + self.offset) < len(cache):
            return

        if self._name in cache[token_index + self.offset]:
            return cache[token_index + self.offset].get(self._name, None)
        else:
            tokens = [c["token"] for c in cache]
            value = self.function(tokens, token_index + self.offset)
            if value is not None:
                cache[token_index + self.offset][self._name] = value
            return value


def crf_features(dataset, intent, language, crf_features_config, random_state):
    if language == Language.EN:
        return en_features(dataset, intent, crf_features_config,
                           random_state=random_state)
    elif language == Language.ES:
        return default_features(language, dataset, intent, crf_features_config,
                                use_stemming=True, random_state=random_state)
    elif language == Language.FR:
        return fr_features(dataset, intent, crf_features_config,
                           random_state=random_state)
    elif language == Language.DE:
        return de_features(dataset, intent, crf_features_config,
                           random_state=random_state)
    elif language == Language.KO:
        return ko_features(dataset, intent, crf_features_config,
                           random_state=random_state)
    else:
        raise NotImplementedError("Feature function are not implemented for "
                                  "%s" % language)


# Base feature functions and factories
def is_digit():
    return Feature(
        "is_digit",
        lambda tokens, token_index: "1" if tokens[
            token_index].value.isdigit() else None
    )


def is_first():
    return Feature(
        "is_first",
        lambda tokens, token_index: "1" if token_index == 0 else None
    )


def is_last():
    return Feature(
        "is_last",
        lambda tokens, token_index: "1" if token_index == len(
            tokens) - 1 else None
    )


def get_prefix_fn(prefix_size):
    def prefix(tokens, token_index):
        return get_word_chunk(tokens[token_index].normalized_value,
                              prefix_size, 0)

    return Feature("prefix-%s" % prefix_size, prefix,
                   feature_type="prefix_%s")


def get_suffix_fn(suffix_size):
    def suffix(tokens, token_index):
        return get_word_chunk(tokens[token_index].normalized_value,
                              suffix_size, len(tokens[token_index].value),
                              reverse=True)

    return Feature("suffix-%s" % suffix_size, suffix,
                   feature_type="suffix_%s")


def get_length_fn():
    return Feature(
        "length", lambda tokens, token_index: len(tokens[token_index].value))


def get_ngram_fn(n, use_stemming, language_code,
                 common_words_gazetteer_name=None):
    language = Language.from_iso_code(language_code)
    if n < 1:
        raise ValueError("n should be >= 1")

    gazetteer = None
    if common_words_gazetteer_name is not None:
        gazetteer = get_gazetteer(language, common_words_gazetteer_name)
        if use_stemming:
            gazetteer = set(stem(w, language) for w in gazetteer)

    def ngram(tokens, token_index):
        max_len = len(tokens)
        end = token_index + n
        if 0 <= token_index < max_len and end <= max_len:
            if gazetteer is None:
                if use_stemming:
                    return language.default_sep.join(
                        t.stem for t in tokens[token_index:end])
                return language.default_sep.join(
                    t.normalized_value for t in tokens[token_index:end])
            words = []
            for t in tokens[token_index:end]:
                normalized = t.stem if use_stemming else t.normalized_value
                words.append(normalized if normalized in gazetteer
                             else "rare_word")
            return language.default_sep.join(words)
        return None

    return Feature("ngram_%s" % n, ngram)


def get_shape_ngram_fn(n, language_code):
    language = Language.from_iso_code(language_code)
    if n < 1:
        raise ValueError("n should be >= 1")

    def shape_ngram(tokens, token_index):
        max_len = len(tokens)
        end = token_index + n
        if 0 <= token_index < max_len and end <= max_len:
            return language.default_sep.join(get_shape(t.value)
                                             for t in tokens[token_index:end])
        return None

    return Feature("shape_ngram_%s" % n, shape_ngram)


def get_word_cluster_fn(cluster_name, language_code, use_stemming):
    language = Language.from_iso_code(language_code)

    def word_cluster(tokens, token_index):
        normalized_value = tokens[token_index].stem if use_stemming \
            else tokens[token_index].normalized_value
        return get_word_clusters(language)[cluster_name].get(normalized_value,
                                                             None)

    return Feature("word_cluster_%s" % cluster_name, word_cluster,
                   feature_type="word_cluster")


# pylint: disable=unused-argument
def get_token_is_in_fn(tokens_collection, collection_name, use_stemming,
                       tagging_scheme_code, language_code):
    tagging_scheme = TaggingScheme(tagging_scheme_code)
    tokens_collection = set(tokens_collection)

    def transform(token):
        return token.stem if use_stemming else token.normalized_value

    def token_is_in(tokens, token_index):
        normalized_tokens = map(transform, tokens)
        ngrams = get_all_ngrams(normalized_tokens)
        ngrams = [ng for ng in ngrams if token_index in ng[TOKEN_INDEXES]]
        ngrams = sorted(ngrams, key=lambda ng: len(ng[TOKEN_INDEXES]),
                        reverse=True)
        for ngram in ngrams:
            if ngram[NGRAM] in tokens_collection:
                return get_scheme_prefix(token_index,
                                         sorted(ngram[TOKEN_INDEXES]),
                                         tagging_scheme)
        return None

    return Feature("token_is_in_%s" % collection_name, token_is_in,
                   feature_type="collection_match")


# pylint: enable=unused-argument


def entity_filter(entity, start, end):
    return (entity[MATCH_RANGE][0] <= start < entity[MATCH_RANGE][1]) and \
           (entity[MATCH_RANGE][0] < end <= entity[MATCH_RANGE][1])


def get_built_in_annotation_fn(built_in_entity_label, language_code,
                               tagging_scheme_code):
    language = Language.from_iso_code(language_code)
    tagging_scheme = TaggingScheme(tagging_scheme_code)
    built_in_entity = BuiltInEntity.from_label(built_in_entity_label)
    feature_name = "built-in-%s" % built_in_entity.value["label"]

    def built_in_annotation(tokens, token_index):
        text = initial_string_from_tokens(tokens)
        start = tokens[token_index].start
        end = tokens[token_index].end

        builtin_entities = get_builtin_entities(
            text, language, scope=[built_in_entity])
        builtin_entities = [ent for ent in builtin_entities
                            if entity_filter(ent, start, end)]
        for ent in builtin_entities:
            entity_start = ent[MATCH_RANGE][0]
            entity_end = ent[MATCH_RANGE][1]
            indexes = []
            for index, token in enumerate(tokens):
                if (entity_start <= token.start < entity_end) \
                        and (entity_start < token.end <= entity_end):
                    indexes.append(index)
            return get_scheme_prefix(token_index, indexes, tagging_scheme)

    return Feature(feature_name, built_in_annotation,
                   feature_type="builtin_match")
