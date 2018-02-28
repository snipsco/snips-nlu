from __future__ import unicode_literals

import glob
import io
import os
from collections import defaultdict

from builtins import next
from future.utils import iteritems
from snips_nlu_utils import normalize

from snips_nlu.constants import (STOP_WORDS, WORD_CLUSTERS, GAZETTEERS, NOISE,
                                 RESOURCES_PATH, LANGUAGE_EN, LANGUAGE_FR,
                                 LANGUAGE_ES, LANGUAGE_KO, LANGUAGE_DE,
                                 LANGUAGE_JA, STEMS)
from snips_nlu.languages import get_ignored_characters_pattern
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

_RESOURCES = defaultdict(dict)


class UnknownResource(LookupError):
    pass


class UnloadedResources(LookupError):
    pass


def get_language_resource(language):
    if language not in _RESOURCES:
        raise UnloadedResources(
            "Missing resources for '{}', please load them with the "
            "load_resources function".format(language))
    return _RESOURCES[language]


def get_resource(language, resource_name):
    language_resource = get_language_resource(language)
    try:
        return language_resource[resource_name]
    except KeyError:
        raise UnknownResource("Unknown resource '{}' for '{}' "
                              "language".format(resource_name, language))


def _load_stop_words(language):
    if STOP_WORDS in RESOURCE_INDEX[language]:
        stop_words_file_path = os.path.join(
            get_resources_path(language),
            RESOURCE_INDEX[language][STOP_WORDS])
        with io.open(stop_words_file_path, encoding='utf8') as f:
            lines = (normalize(l) for l in f)
            _RESOURCES[language][STOP_WORDS] = set(l for l in lines if l)


def get_stop_words(language):
    return get_resource(language, STOP_WORDS)


def _load_noises(language):
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
            _RESOURCES[language][NOISE] = next(f).split()


def get_noises(language):
    return get_resource(language, NOISE)


def _load_clusters(language):
    word_clusters_paths = {
        os.path.splitext(name)[0]: os.path.join(
            get_resources_path(language), name)
        for name in RESOURCE_INDEX[language].get(WORD_CLUSTERS, [])
    }
    if WORD_CLUSTERS in RESOURCE_INDEX[language]:
        clusters = dict()
        for name, path in iteritems(word_clusters_paths):
            with io.open(path, encoding="utf8") as f:
                clusters[name] = dict()
                for l in f:
                    split = l.rstrip().split("\t")
                    if len(split) == 2:
                        clusters[name][split[0]] = split[1]
        _RESOURCES[language][WORD_CLUSTERS] = clusters


def get_word_clusters(language):
    return get_resource(language, WORD_CLUSTERS)


def _load_gazetteers(language):
    gazetteers_paths = {
        os.path.splitext(name)[0]: os.path.join(
            get_resources_path(language), name)
        for name in RESOURCE_INDEX[language].get(GAZETTEERS, [])
    }
    gazetteers = dict()
    for name, path in iteritems(gazetteers_paths):
        with io.open(path, encoding="utf8") as f:
            gazetteers[name] = set()
            for l in f:
                normalized = normalize(l.strip())
                if normalized:
                    normalized = get_ignored_characters_pattern(language).join(
                        [t.value for t in tokenize(normalized, language)])
                    gazetteers[name].add(normalized)
    _RESOURCES[language][GAZETTEERS] = gazetteers


def get_gazetteers(language):
    return get_resource(language, GAZETTEERS)


def get_gazetteer(language, gazetteer_name):
    return get_gazetteers(language)[gazetteer_name]


def _verbs_lexemes(language):
    stems_paths = glob.glob(os.path.join(RESOURCES_PATH, language,
                                         "top_*_verbs_lexemes.txt"))
    if not stems_paths:
        return dict()

    verb_lexemes = dict()
    with io.open(stems_paths[0], encoding="utf8") as f:
        for line in f:
            elements = line.strip().split(';')
            verb = normalize(elements[0])
            lexemes = elements[1].split(',')
            verb_lexemes.update(
                {normalize(lexeme): verb for lexeme in lexemes})
    return verb_lexemes


def _word_inflections(language):
    inflection_paths = glob.glob(os.path.join(RESOURCES_PATH, language,
                                              "top_*_words_inflected.txt"))
    if not inflection_paths:
        return dict()

    inflections = dict()
    with io.open(inflection_paths[0], encoding="utf8") as f:
        for line in f:
            elements = line.strip().split(';')
            inflections[normalize(elements[0])] = normalize(elements[1])
    return inflections


def _load_stems(language):
    stems = _word_inflections(language)
    stems.update(_verbs_lexemes(language))
    _RESOURCES[language][STEMS] = stems


def get_stems(language):
    return get_resource(language, STEMS)


def load_resources(language):
    """Load language specific resources

    Args:
        language (str): language
    """
    if language in _RESOURCES:
        return
    _load_clusters(language)
    _load_gazetteers(language)
    _load_stop_words(language)
    _load_noises(language)
    _load_stems(language)
