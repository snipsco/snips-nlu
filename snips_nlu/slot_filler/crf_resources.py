import io
import os
import re

from snips_nlu.languages import Language
from snips_nlu.tokenization import tokenize
from snips_nlu.utils import get_resources_path

CLUSTER_NAMES = {
    Language.EN: ["brown_clusters"]
}

WORD_CLUSTERS = dict()


def get_word_clusters(language):
    global WORD_CLUSTERS
    word_clusters_paths = {
        name: os.path.join(get_resources_path(language), "%s.txt" % name)
        for name in CLUSTER_NAMES.get(language, [])
    }
    if language not in WORD_CLUSTERS:
        _word_clusters = dict()
        WORD_CLUSTERS[language] = _word_clusters
        for name, path in word_clusters_paths.iteritems():
            with io.open(path, encoding="utf8") as f:
                _word_clusters[name] = dict()
                for l in f:
                    split = l.rstrip().lower().split("\t")
                    normalized = " ".join(
                        [t.value for t in tokenize(split[0])])
                    if len(split) == 2:
                        _word_clusters[name][normalized] = split[1]

    return WORD_CLUSTERS[language]


GAZETTEERS_NAMES = {
    Language.EN: ["top_10000_nouns", "cities_us", "cities_world",
                  "countries", "states_us", "stop_words",
                  "street_identifier", "top_10000_words"]
}

GAZETTEERS = dict()


def get_gazetteers(language):
    global GAZETTEERS
    gazetteers_paths = {
        name: os.path.join(get_resources_path(language), "%s.txt" % name)
        for name in GAZETTEERS_NAMES.get(language, [])
    }
    if language not in GAZETTEERS:
        _gazetteers = dict()
        GAZETTEERS[language] = _gazetteers
        for name, path in gazetteers_paths.iteritems():
            with io.open(path, encoding="utf8") as f:
                _gazetteers[name] = set()
                for l in f:
                    normalized = l.strip().lower()
                    if len(normalized) > 0:
                        normalized = " ".join(
                            [t.value for t in tokenize(normalized)])
                        _gazetteers[name].add(normalized)

    return GAZETTEERS[language]


def get_gazetteer(language, gazetteer_name):
    return get_gazetteers(language)[gazetteer_name]


GAZETTEERS_REGEXES = dict()


def get_gazetteers_regexes(language):
    global GAZETTEERS_REGEXES
    if language not in GAZETTEERS_REGEXES:
        gazetteers = get_gazetteers(language)
        _gazetteers_regexes = dict()
        GAZETTEERS_REGEXES[language] = _gazetteers_regexes
        for name, expression_set in gazetteers.iteritems():
            pattern = r"|".join(re.escape(e) for e in
                                sorted(expression_set, key=len, reverse=True))
            regex = re.compile(pattern, re.IGNORECASE)
            _gazetteers_regexes[name] = regex
    return GAZETTEERS_REGEXES[language]


def get_gazetteer_regex(language, gazetteer_name):
    return get_gazetteers_regexes(language)[gazetteer_name]
