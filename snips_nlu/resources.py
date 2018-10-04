from __future__ import unicode_literals

import json
import shutil
from builtins import next
from pathlib import Path

from snips_nlu.constants import (
    CUSTOM_ENTITY_PARSER_USAGE, DATA_PATH, GAZETTEERS, NOISE, RESOURCES_DIR,
    STEMS, STOP_WORDS, WORD_CLUSTERS)
from snips_nlu.entity_parser.custom_entity_parser import (
    CustomEntityParserUsage)
from snips_nlu.utils import get_package_path, is_package, json_string

_RESOURCES = dict()


class MissingResource(LookupError):
    pass


def clear_resources():
    _RESOURCES.clear()


def load_resources(name):
    """Load language specific resources

    Args:
        name (str): Resource name as in ``snips-nlu download <name>``. Can also
            be the name of a python package or a directory path.

    Note:
        Language resources must be loaded before fitting or parsing
    """
    if name in set(d.name for d in DATA_PATH.iterdir()):
        load_resources_from_dir(DATA_PATH / name)
    elif is_package(name):
        package_path = get_package_path(name)
        resources_sub_dir = get_resources_sub_directory(package_path)
        load_resources_from_dir(resources_sub_dir)
    elif Path(name).exists():
        path = Path(name)
        if (path / "__init__.py").exists():
            path = get_resources_sub_directory(path)
        load_resources_from_dir(path)
    else:
        raise MissingResource("Language resource '{r}' not found. This may be "
                              "solved by running "
                              "'python -m snips_nlu download {r}'"
                              .format(r=name))


def load_resources_from_dir(resources_dir):
    with (resources_dir / "metadata.json").open(encoding="utf8") as f:
        metadata = json.load(f)
    language = metadata["language"]
    if language in _RESOURCES:
        return

    try:
        gazetteer_names = metadata["gazetteers"]
        clusters_names = metadata["word_clusters"]
        stop_words_filename = metadata["stop_words"]
        stems_filename = metadata["stems"]
        noise_filename = metadata["noise"]
    except KeyError:
        print_compatibility_error(language)
        raise

    gazetteers = _load_gazetteers(resources_dir / "gazetteers",
                                  gazetteer_names)
    stems = _load_stems(resources_dir / "stemming", stems_filename)
    word_clusters = _load_word_clusters(resources_dir / "word_clusters",
                                        clusters_names)
    stop_words = _load_stop_words(resources_dir, stop_words_filename)
    noise = _load_noise(resources_dir, noise_filename)

    _RESOURCES[language] = {
        WORD_CLUSTERS: word_clusters,
        GAZETTEERS: gazetteers,
        STOP_WORDS: stop_words,
        NOISE: noise,
        STEMS: stems,
        RESOURCES_DIR: str(resources_dir),
    }


def print_compatibility_error(language):
    from snips_nlu.cli.utils import PrettyPrintLevel, pretty_print
    pretty_print(
        "Language resources for '{lang}' could not be loaded.\nYou may "
        "have to download resources again using "
        "'python -m snips_nlu download {lang}'".format(lang=language),
        "This can happen when you update the snips-nlu library.",
        title="Something went wrong while loading resources",
        level=PrettyPrintLevel.ERROR)


def get_resources_sub_directory(resources_dir):
    resources_dir = Path(resources_dir)
    with (resources_dir / "metadata.json").open(encoding="utf8") as f:
        metadata = json.load(f)
    resource_name = metadata["name"]
    version = metadata["version"]
    sub_dir_name = "{r}-{v}".format(r=resource_name, v=version)
    return resources_dir / sub_dir_name


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


def get_resources_dir(language):
    return _get_resource(language, RESOURCES_DIR)


def merge_required_resources(lhs, rhs):
    if not lhs:
        return dict() if rhs is None else rhs
    if not rhs:
        return dict() if lhs is None else lhs
    merged_resources = dict()
    if lhs.get(NOISE, False) or rhs.get(NOISE, False):
        merged_resources[NOISE] = True
    if lhs.get(STOP_WORDS, False) or rhs.get(STOP_WORDS, False):
        merged_resources[STOP_WORDS] = True
    if lhs.get(STEMS, False) or rhs.get(STEMS, False):
        merged_resources[STEMS] = True
    lhs_parser_usage = lhs.get(CUSTOM_ENTITY_PARSER_USAGE)
    rhs_parser_usage = rhs.get(CUSTOM_ENTITY_PARSER_USAGE)
    parser_usage = CustomEntityParserUsage.merge_usages(
        lhs_parser_usage, rhs_parser_usage)
    merged_resources[CUSTOM_ENTITY_PARSER_USAGE] = parser_usage
    gazetteers = lhs.get(GAZETTEERS, set()).union(rhs.get(GAZETTEERS, set()))
    if gazetteers:
        merged_resources[GAZETTEERS] = gazetteers
    word_clusters = lhs.get(WORD_CLUSTERS, set()).union(
        rhs.get(WORD_CLUSTERS, set()))
    if word_clusters:
        merged_resources[WORD_CLUSTERS] = word_clusters
    return merged_resources


def persist_resources(resources_dest_path, required_resources, language):
    if not required_resources:
        return

    resources_dest_path.mkdir()

    resources_src_path = Path(get_resources_dir(language))
    with (resources_src_path / "metadata.json").open(encoding="utf8") as f:
        metadata = json.load(f)

    # Update metadata and keep only required resources
    if not required_resources.get(NOISE, False):
        metadata[NOISE] = None
    if not required_resources.get(STOP_WORDS, False):
        metadata[STOP_WORDS] = None
    if not required_resources.get(STEMS, False):
        metadata[STEMS] = None

    metadata[GAZETTEERS] = sorted(required_resources.get(GAZETTEERS, []))
    metadata[WORD_CLUSTERS] = sorted(required_resources.get(WORD_CLUSTERS, []))
    metadata_dest_path = resources_dest_path / "metadata.json"
    metadata_json = json_string(metadata)
    with metadata_dest_path.open(encoding="utf8", mode="w") as f:
        f.write(metadata_json)

    if metadata[NOISE] is not None:
        noise_src = (resources_src_path / metadata[NOISE]).with_suffix(".txt")
        noise_dest = (resources_dest_path / noise_src.name)
        shutil.copy(str(noise_src), str(noise_dest))

    if metadata[STOP_WORDS] is not None:
        stop_words_src = (resources_src_path / metadata[STOP_WORDS]) \
            .with_suffix(".txt")
        stop_words_dest = (resources_dest_path / stop_words_src.name)
        shutil.copy(str(stop_words_src), str(stop_words_dest))

    if metadata[STEMS] is not None:
        stems_src = (resources_src_path / "stemming" / metadata["stems"]) \
            .with_suffix(".txt")
        stemming_dir = resources_dest_path / "stemming"
        stemming_dir.mkdir()
        stems_dest = stemming_dir / stems_src.name
        shutil.copy(str(stems_src), str(stems_dest))

    if metadata[GAZETTEERS]:
        gazetteer_src_dir = resources_src_path / "gazetteers"
        gazetteer_dest_dir = resources_dest_path / "gazetteers"
        gazetteer_dest_dir.mkdir()
        for gazetteer in metadata[GAZETTEERS]:
            gazetteer_src = (gazetteer_src_dir / gazetteer) \
                .with_suffix(".txt")
            gazetteer_dest = gazetteer_dest_dir / gazetteer_src.name
            shutil.copy(str(gazetteer_src), str(gazetteer_dest))

    if metadata[WORD_CLUSTERS]:
        clusters_src_dir = resources_src_path / "word_clusters"
        clusters_dest_dir = resources_dest_path / "word_clusters"
        clusters_dest_dir.mkdir()
        for word_clusters in metadata["word_clusters"]:
            clusters_src = (clusters_src_dir / word_clusters) \
                .with_suffix(".txt")
            clusters_dest = clusters_dest_dir / clusters_src.name
            shutil.copy(str(clusters_src), str(clusters_dest))


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


def _load_stop_words(resources_dir, stop_words_filename):
    if not stop_words_filename:
        return None
    stop_words_path = (resources_dir / stop_words_filename).with_suffix(".txt")
    with stop_words_path.open(encoding='utf8') as f:
        stop_words = set(l.strip() for l in f)
    return stop_words


def _load_noise(resources_dir, noise_filename):
    if not noise_filename:
        return None
    noise_path = (resources_dir / noise_filename).with_suffix(".txt")
    with noise_path.open(encoding='utf8') as f:
        # Here we split on a " " knowing that it's always ignored by
        # the tokenization (see tokenization unit tests)
        # It is not important to tokenize precisely as this noise is just used
        # to generate utterances for the None intent
        noise = next(f).split()
    return noise


def _load_word_clusters(word_clusters_dir, clusters_names):
    if not clusters_names:
        return dict()

    clusters = dict()
    for clusters_name in clusters_names:
        clusters_path = (word_clusters_dir / clusters_name).with_suffix(".txt")
        clusters[clusters_name] = dict()
        with clusters_path.open(encoding="utf8") as f:
            for line in f:
                split = line.rstrip().split("\t")
                clusters[clusters_name][split[0]] = split[1]
    return clusters


def _load_gazetteers(gazetteers_dir, gazetteer_names):
    if not gazetteer_names:
        return dict()

    gazetteers = dict()
    for gazetteer_name in gazetteer_names:
        gazetteer_path = (gazetteers_dir / gazetteer_name).with_suffix(".txt")
        with gazetteer_path.open(encoding="utf8") as f:
            gazetteers[gazetteer_name] = set(v.strip() for v in f)
    return gazetteers


def _load_stems(stems_dir, filename):
    if not filename:
        return None
    stems_path = (stems_dir / filename).with_suffix(".txt")
    stems = dict()
    with stems_path.open(encoding="utf8") as f:
        for line in f:
            elements = line.strip().split(',')
            stem = elements[0]
            for value in elements[1:]:
                stems[value] = stem
    return stems
