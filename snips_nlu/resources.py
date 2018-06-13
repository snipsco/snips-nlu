from __future__ import unicode_literals

import glob
import io
import os
from builtins import next
from collections import defaultdict

from snips_nlu_ontology import get_all_languages
from snips_nlu_utils import normalize

from snips_nlu.constants import (STOP_WORDS, WORD_CLUSTERS, GAZETTEERS, NOISE,
                                 STEMS)
from snips_nlu.languages import get_default_sep
from snips_nlu.tokenization import tokenize

_RESOURCES = defaultdict(dict)


class MissingResource(LookupError):
    pass


def clear_resources():
    _RESOURCES.clear()


def load_resources(resources_dir):
    """Load language specific resources

    Args:
        resources_dir (str): resources directory path

    Note:
        Language resources must be loaded before fitting or parsing
    """
    clear_resources()
    languages = os.listdir(resources_dir)
    all_supported_languages = get_all_languages()
    for language in languages:
        if language not in all_supported_languages:
            raise ValueError("Unknown language: '%s'" % language)
        _RESOURCES[language] = _load_resources(resources_dir, language)


def resource_exists(language, resource_name):
    """Tell if the resource specified by the resource_name exist

        Args:
            language (str): language
            resource_name (str): the resource name
        Returns:
            bool: whether the resource exists or not
    """
    return resource_name in _RESOURCES[language]


def get_stop_words(language):
    return _get_resource(language, STOP_WORDS)


def get_noise(language):
    return _get_resource(language, NOISE)


def get_word_clusters(language):
    return _get_resource(language, WORD_CLUSTERS)


def get_word_cluster(language, cluster_name):
    word_clusters = get_word_clusters(language)
    if cluster_name not in word_clusters:
        raise MissingResource("Word cluster '{}' not found for language '{}'"
                              .format(cluster_name, language))
    return word_clusters[cluster_name]


def get_gazetteer(language, gazetteer_name):
    gazetteers = _get_resource(language, GAZETTEERS)
    if gazetteer_name not in gazetteers:
        raise MissingResource("Gazetteer '{}' not found for language '{}'"
                              .format(gazetteer_name, language))
    return gazetteers[gazetteer_name]


def get_stems(language):
    return _get_resource(language, STEMS)


def _get_resource(language, resource_name):
    if language not in _RESOURCES:
        raise MissingResource(
            "Missing resources for language '%s', please load them with the "
            "load_resources function" % language)
    if resource_name not in _RESOURCES[language] \
            or _RESOURCES[language][resource_name] is None:
        raise MissingResource("Resource '{}' not found for language '{}'"
                              .format(resource_name, language))
    return _RESOURCES[language][resource_name]


def _load_resources(resources_dir, language):
    language_resources_path = os.path.join(resources_dir, language)
    word_clusters = _load_word_clusters(
        os.path.join(language_resources_path, "word_clusters"))
    gazetteers = _load_gazetteers(language_resources_path, language)
    stop_words = _load_stop_words(
        os.path.join(language_resources_path, "stop_words.txt"))
    noise = _load_noise(os.path.join(language_resources_path, "noise.txt"))
    stems = _load_stems(os.path.join(language_resources_path, "stemming"))

    return {
        WORD_CLUSTERS: word_clusters,
        GAZETTEERS: gazetteers,
        STOP_WORDS: stop_words,
        NOISE: noise,
        STEMS: stems
    }


def _load_stop_words(stop_words_path):
    if not os.path.exists(stop_words_path):
        return None
    with io.open(stop_words_path, encoding='utf8') as f:
        lines = (normalize(l) for l in f)
    stop_words = set(l for l in lines if l)
    return stop_words


def _load_noise(noise_path):
    if not os.path.exists(noise_path):
        return None
    with io.open(noise_path, encoding='utf8') as f:
        # Here we split on a " " knowing that it's always ignored by
        # the tokenization (see tokenization unit tests)
        # It is not important to tokenize precisely as this noise is just used
        # to generate utterances for the None intent
        noise = next(f).split()
    return noise


def _load_word_clusters(word_clusters_path):
    if not os.path.isdir(word_clusters_path):
        return dict()

    clusters = dict()
    for filename in os.listdir(word_clusters_path):
        word_cluster_name = os.path.splitext(filename)[0]
        word_cluster_path = os.path.join(word_clusters_path, filename)
        with io.open(word_cluster_path, encoding="utf8") as f:
            clusters[word_cluster_name] = dict()
            for line in f:
                split = line.rstrip().split("\t")
                if len(split) == 2:
                    clusters[word_cluster_name][split[0]] = split[1]
    return clusters


def _load_gazetteers(language_resources_path, language):
    gazetteers_path = os.path.join(language_resources_path, "gazetteers")
    if not os.path.isdir(gazetteers_path):
        return dict()

    gazetteers = dict()
    for filename in os.listdir(gazetteers_path):
        gazetteer_name = os.path.splitext(filename)[0]
        gazetteer_path = os.path.join(gazetteers_path, filename)
        with io.open(gazetteer_path, encoding="utf8") as f:
            gazetteers[gazetteer_name] = set()
            for line in f:
                normalized = normalize(line.strip())
                if normalized:
                    token_values = (t.value
                                    for t in tokenize(normalized, language))
                    normalized = get_default_sep(language).join(token_values)
                    gazetteers[gazetteer_name].add(normalized)
    return gazetteers


def _load_verbs_lexemes(stemming_path):
    lexems_paths = glob.glob(
        os.path.join(stemming_path, "top_*_verbs_lexemes.txt"))
    if not lexems_paths:
        return None

    verb_lexemes = dict()
    with io.open(lexems_paths[0], encoding="utf8") as f:
        for line in f:
            elements = line.strip().split(';')
            verb = normalize(elements[0])
            lexemes = elements[1].split(',')
            verb_lexemes.update(
                {normalize(lexeme): verb for lexeme in lexemes})
    return verb_lexemes


def _load_words_inflections(stemming_path):
    inflection_paths = glob.glob(
        os.path.join(stemming_path, "top_*_words_inflected.txt"))
    if not inflection_paths:
        return None

    inflections = dict()
    with io.open(inflection_paths[0], encoding="utf8") as f:
        for line in f:
            elements = line.strip().split(';')
            inflections[normalize(elements[0])] = normalize(elements[1])
    return inflections


def _load_stems(stemming_path):
    inflexions = _load_words_inflections(stemming_path)
    lexemes = _load_verbs_lexemes(stemming_path)
    stems = None
    if inflexions is not None:
        stems = inflexions
    if lexemes is not None:
        if stems is None:
            stems = dict()
        stems.update(lexemes)
    return stems
