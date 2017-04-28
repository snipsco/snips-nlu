from __future__ import unicode_literals

import io
import os

from snips_nlu.constants import (STOP_WORDS, SUBTITLES,
                                 WORD_CLUSTERS, GAZETTEERS)
from snips_nlu.languages import Language
from snips_nlu.tokenization import tokenize
from snips_nlu.utils import get_resources_path

RESOURCE_INDEX = {
    Language.EN: {
        GAZETTEERS: ["top_10000_nouns.txt", "cities_us.txt",
                     "cities_world.txt", "countries.txt", "states_us.txt",
                     "stop_words.txt", "street_identifier.txt",
                     "top_10000_words.txt"],
        STOP_WORDS: "stop_words.txt",
        SUBTITLES: "subtitles.txt",
        WORD_CLUSTERS: ["brown_clusters.txt"]
    },
    Language.FR: {
        STOP_WORDS: "stop_words.txt",
        SUBTITLES: "subtitles.txt",
    },
    Language.ES: {
        STOP_WORDS: "stop_words.txt",
        SUBTITLES: "subtitles.txt",
    },
    Language.KO: {
        STOP_WORDS: "stop_words.txt",
        SUBTITLES: "subtitles.txt",
    },
    Language.DE: {
        STOP_WORDS: "stop_words.txt",
        SUBTITLES: "subtitles.txt",
    }
}

_STOP_WORDS = dict()
_SUBTITLES = dict()
_GAZETTEERS = dict()
_WORD_CLUSTERS = dict()
_GAZETTEERS_REGEXES = dict()


def load_stop_words():
    for language in Language:
        if STOP_WORDS in RESOURCE_INDEX[language]:
            stop_words_file_path = os.path.join(
                get_resources_path(language),
                RESOURCE_INDEX[language][STOP_WORDS])
            with io.open(stop_words_file_path, encoding='utf8') as f:
                lines = [l.strip() for l in f]
                _STOP_WORDS[language] = set(l for l in lines if len(l) > 0)


def get_stop_words(language):
    return _STOP_WORDS[language]


def load_subtitles():
    for language in Language:
        if SUBTITLES in RESOURCE_INDEX[language]:
            subtitles_file_path = os.path.join(
                get_resources_path(language),
                RESOURCE_INDEX[language][SUBTITLES])
            with io.open(subtitles_file_path, encoding='utf8') as f:
                lines = [l.strip() for l in f]
            _SUBTITLES[language] = set(l for l in lines if len(l) > 0)


def get_subtitles(language):
    return _SUBTITLES[language]


def load_clusters():
    for language in Language:
        word_clusters_paths = {
            os.path.splitext(name)[0]: os.path.join(
                get_resources_path(language), name)
            for name in RESOURCE_INDEX[language].get(WORD_CLUSTERS, [])
        }
        if WORD_CLUSTERS in RESOURCE_INDEX[language]:
            _word_clusters = dict()
            _WORD_CLUSTERS[language] = _word_clusters
            for name, path in word_clusters_paths.iteritems():
                with io.open(path, encoding="utf8") as f:
                    _word_clusters[name] = dict()
                    for l in f:
                        split = l.rstrip().lower().split("\t")
                        normalized = " ".join(
                            [t.value for t in tokenize(split[0])])
                        if len(split) == 2:
                            _word_clusters[name][normalized] = split[1]


def get_word_clusters(language):
    return _WORD_CLUSTERS[language]


def load_gazetteers():
    for language in Language:
        gazetteers_paths = {
            os.path.splitext(name)[0]: os.path.join(
                get_resources_path(language), name)
            for name in RESOURCE_INDEX[language].get(GAZETTEERS, [])
        }
        _gazetteers = dict()
        _GAZETTEERS[language] = _gazetteers
        for name, path in gazetteers_paths.iteritems():
            with io.open(path, encoding="utf8") as f:
                _gazetteers[name] = set()
                for l in f:
                    normalized = l.strip().lower()
                    if len(normalized) > 0:
                        normalized = " ".join(
                            [t.value for t in tokenize(normalized)])
                        _gazetteers[name].add(normalized)


def get_gazetteers(language):
    return _GAZETTEERS[language]


def get_gazetteer(language, gazetteer_name):
    return get_gazetteers(language)[gazetteer_name]


def load_resources():
    load_clusters()
    load_gazetteers()
    load_stop_words()
    load_subtitles()
