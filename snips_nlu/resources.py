from __future__ import unicode_literals

import json
from builtins import next
from pathlib import Path

from snips_nlu_utils import normalize

from snips_nlu.constants import (STOP_WORDS, WORD_CLUSTERS, GAZETTEERS, NOISE,
                                 STEMS, DATA_PATH)
from snips_nlu.languages import get_default_sep
from snips_nlu.tokenization import tokenize
from snips_nlu.utils import is_package, get_package_path

_RESOURCES = dict()


class MissingResource(LookupError):
    pass


def clear_resources():
    _RESOURCES.clear()


def load_resources(name):
    """Load language specific resources

    Args:
        name (str): resource name

    Note:
        Language resources must be loaded before fitting or parsing
    """
    if name in set(d.name for d in DATA_PATH.iterdir()):
        _load_resources_from_dir(DATA_PATH / name)
    elif is_package(name):
        package_path = get_package_path(name)
        _load_resources_from_dir(package_path)
    elif Path(name).exists():
        _load_resources_from_dir(Path(name))
    else:
        raise MissingResource("Language resource '{r}' not found. This may be "
                              "solved by running 'snips-nlu download {r}'"
                              .format(r=name))


def resource_exists(language, resource_name):
    """Tell if the resource specified by the resource_name exist

        Args:
            language (str): language
            resource_name (str): the resource name
        Returns:
            bool: whether the resource exists or not
    """
    return resource_name in _RESOURCES[language] \
           and _RESOURCES[language][resource_name] is not None


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


def _load_resources_from_dir(resources_dir):
    with (resources_dir / "metadata.json").open() as f:
        metadata = json.load(f)
    language = metadata["language"]
    resource_name = metadata["name"]
    version = metadata["version"]
    if language in _RESOURCES:
        return
    sub_dir = resources_dir / (resource_name + "-" + version)
    if not sub_dir.is_dir():
        raise FileNotFoundError("Missing resources directory: %s"
                                % str(sub_dir))
    word_clusters = _load_word_clusters(sub_dir / "word_clusters")
    gazetteers = _load_gazetteers(sub_dir / "gazetteers", language)
    stop_words = _load_stop_words(sub_dir / "stop_words.txt")
    noise = _load_noise(sub_dir / "noise.txt")
    stems = _load_stems(sub_dir / "stemming")

    _RESOURCES[language] = {
        WORD_CLUSTERS: word_clusters,
        GAZETTEERS: gazetteers,
        STOP_WORDS: stop_words,
        NOISE: noise,
        STEMS: stems
    }


def _load_stop_words(stop_words_path):
    if not stop_words_path.exists():
        return None
    with stop_words_path.open(encoding='utf8') as f:
        lines = (normalize(l) for l in f)
        stop_words = set(l for l in lines if l)
    return stop_words


def _load_noise(noise_path):
    if not noise_path.exists():
        return None
    with noise_path.open(encoding='utf8') as f:
        # Here we split on a " " knowing that it's always ignored by
        # the tokenization (see tokenization unit tests)
        # It is not important to tokenize precisely as this noise is just used
        # to generate utterances for the None intent
        noise = next(f).split()
    return noise


def _load_word_clusters(word_clusters_path):
    if not word_clusters_path.is_dir():
        return dict()

    clusters = dict()
    for filepath in word_clusters_path.iterdir():
        word_cluster_name = filepath.stem
        with filepath.open(encoding="utf8") as f:
            clusters[word_cluster_name] = dict()
            for line in f:
                split = line.rstrip().split("\t")
                if len(split) == 2:
                    clusters[word_cluster_name][split[0]] = split[1]
    return clusters


def _load_gazetteers(gazetteers_path, language):
    if not gazetteers_path.is_dir():
        return dict()

    gazetteers = dict()
    for filepath in gazetteers_path.iterdir():
        gazetteer_name = filepath.stem
        with filepath.open(encoding="utf8") as f:
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
    try:
        lexems_path = next(stemming_path.glob("top_*_verbs_lexemes.txt"))
    except StopIteration:
        return None

    verb_lexemes = dict()
    with lexems_path.open(encoding="utf8") as f:
        for line in f:
            elements = line.strip().split(';')
            verb = normalize(elements[0])
            lexemes = elements[1].split(',')
            verb_lexemes.update(
                {normalize(lexeme): verb for lexeme in lexemes})
    return verb_lexemes


def _load_words_inflections(stemming_path):
    try:
        inflection_path = next(stemming_path.glob("top_*_words_inflected.txt"))
    except StopIteration:
        return None

    inflections = dict()
    with inflection_path.open(encoding="utf8") as f:
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
