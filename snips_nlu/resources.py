from __future__ import unicode_literals

import glob
import io
import os
from builtins import next

from future.utils import iteritems
from snips_nlu_utils import normalize
from snips_nlu_ontology import get_all_languages

from snips_nlu.constants import (STOP_WORDS, WORD_CLUSTERS, GAZETTEERS, NOISE,
                                 RESOURCES_PATH, LANGUAGE_EN, LANGUAGE_FR,
                                 LANGUAGE_ES, LANGUAGE_KO, LANGUAGE_DE,
                                 LANGUAGE_JA)
from snips_nlu.languages import get_default_sep
from snips_nlu.tokenization import tokenize
from snips_nlu.utils import get_resources_path

RESOURCE_INDEX = {
    LANGUAGE_DE: {
        GAZETTEERS: [
            "top_10000_words.txt"
        ],
        STOP_WORDS: "stop_words.txt",
        NOISE: "noise.txt",
    },
    LANGUAGE_EN: {
        GAZETTEERS: [
            "top_10000_words.txt"
        ],
        STOP_WORDS: "stop_words.txt",
        NOISE: "noise.txt",
        WORD_CLUSTERS: ["brown_clusters.txt"]
    },
    LANGUAGE_ES: {
        GAZETTEERS: [
            "top_10000_words.txt"
        ],
        STOP_WORDS: "stop_words.txt",
        NOISE: "noise.txt",
    },
    LANGUAGE_FR: {
        GAZETTEERS: [
            "top_10000_words.txt"
        ],
        STOP_WORDS: "stop_words.txt",
        NOISE: "noise.txt",
    },
    LANGUAGE_JA: {
        STOP_WORDS: "stop_words.txt",
        NOISE: "noise.txt",
    },
    LANGUAGE_KO: {
        STOP_WORDS: "stop_words.txt",
        NOISE: "noise.txt",
    },
}

_STOP_WORDS = dict()
_NOISES = dict()
_GAZETTEERS = dict()
_WORD_CLUSTERS = dict()
_GAZETTEERS_REGEXES = dict()
_LANGUAGE_STEMS = dict()


def load_stop_words():
    for language in get_all_languages():
        if STOP_WORDS in RESOURCE_INDEX[language]:
            stop_words_file_path = os.path.join(
                get_resources_path(language),
                RESOURCE_INDEX[language][STOP_WORDS])
            with io.open(stop_words_file_path, encoding='utf8') as f:
                lines = [normalize(l) for l in f]
                _STOP_WORDS[language] = set(l for l in lines if l)


def get_stop_words(language):
    return _STOP_WORDS[language]


def load_noises():
    for language in get_all_languages():
        if NOISE in RESOURCE_INDEX[language]:
            noise_path = os.path.join(
                get_resources_path(language),
                RESOURCE_INDEX[language][NOISE])
            with io.open(noise_path, encoding='utf8') as f:
                # Here we split on a " " knowing that it's always ignored by
                # the tokenization, see tokenization unit tests.
                # We don't really care about tokenizing precisely as this noise
                #  is just used to generate fake query that will be
                # re-tokenized
                _NOISES[language] = next(f).split()


def get_noises(language):
    return _NOISES[language]


def load_clusters():
    for language in get_all_languages():
        word_clusters_paths = {
            os.path.splitext(name)[0]: os.path.join(
                get_resources_path(language), name)
            for name in RESOURCE_INDEX[language].get(WORD_CLUSTERS, [])
        }
        if WORD_CLUSTERS in RESOURCE_INDEX[language]:
            clusters = dict()
            _WORD_CLUSTERS[language] = clusters
            for name, path in iteritems(word_clusters_paths):
                with io.open(path, encoding="utf8") as f:
                    clusters[name] = dict()
                    for l in f:
                        split = l.rstrip().split("\t")
                        if len(split) == 2:
                            clusters[name][split[0]] = split[1]


def get_word_clusters(language):
    return _WORD_CLUSTERS[language]


def load_gazetteers():
    for language in get_all_languages():
        gazetteers_paths = {
            os.path.splitext(name)[0]: os.path.join(
                get_resources_path(language), name)
            for name in RESOURCE_INDEX[language].get(GAZETTEERS, [])
        }
        gazetteers = dict()
        _GAZETTEERS[language] = gazetteers
        for name, path in iteritems(gazetteers_paths):
            with io.open(path, encoding="utf8") as f:
                gazetteers[name] = set()
                for l in f:
                    normalized = normalize(l)
                    if normalized:
                        normalized = get_default_sep(language).join(
                            [t.value for t in tokenize(normalized, language)])
                        gazetteers[name].add(normalized)


def get_gazetteers(language):
    return _GAZETTEERS[language]


def get_gazetteer(language, gazetteer_name):
    return get_gazetteers(language)[gazetteer_name]


def verbs_lexemes(language):
    stems_paths = glob.glob(os.path.join(RESOURCES_PATH, language,
                                         "top_*_verbs_lexemes.txt"))
    if not stems_paths:
        return dict()

    verb_lexemes = dict()
    with io.open(stems_paths[0], encoding="utf8") as f:
        lines = [l.strip() for l in f]
    for line in lines:
        elements = line.split(';')
        verb = normalize(elements[0])
        lexemes = elements[1].split(',')
        verb_lexemes.update({normalize(lexeme): verb for lexeme in lexemes})
    return verb_lexemes


def word_inflections(language):
    inflection_paths = glob.glob(os.path.join(RESOURCES_PATH, language,
                                              "top_*_words_inflected.txt"))
    if not inflection_paths:
        return dict()

    inflections = dict()
    with io.open(inflection_paths[0], encoding="utf8") as f:
        lines = [l.strip() for l in f]

    for line in lines:
        elements = line.split(';')
        inflections[normalize(elements[0])] = normalize(elements[1])
    return inflections


def load_stems():
    global _LANGUAGE_STEMS
    for language in get_all_languages():
        _LANGUAGE_STEMS[language] = word_inflections(language)
        _LANGUAGE_STEMS[language].update(verbs_lexemes(language))


def get_stems():
    return _LANGUAGE_STEMS


def load_resources():
    load_clusters()
    load_gazetteers()
    load_stop_words()
    load_noises()
    load_stems()
