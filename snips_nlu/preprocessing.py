import glob
import io
import os

from snips_nlu.utils import RESOURCES_PATH

_LANGUAGE_STEMS = dict()


def verbs_stems(language):
    stems_paths = glob.glob(os.path.join(RESOURCES_PATH, language.iso_code,
                                         "top_*_verbs_conjugated.txt"))
    if len(stems_paths) == 0:
        return dict()

    verb_stemmings = dict()
    with io.open(stems_paths[0], encoding="utf8") as f:
        lines = [l.strip() for l in f]
    for line in lines:
        elements = line.split(';')
        verb_stemmings.update(
            {inflection.split(',')[1]: elements[0] for inflection in
             elements[1:]})
    return verb_stemmings


def verbs_lexemes(language):
    stems_paths = glob.glob(os.path.join(RESOURCES_PATH, language.iso_code,
                                         "top_*_verbs_lexemes.txt"))
    if len(stems_paths) == 0:
        return dict()

    verb_lexemes = dict()
    with io.open(stems_paths[0], encoding="utf8") as f:
        lines = [l.strip() for l in f]
    for line in lines:
        elements = line.split(';')
        verb = elements[0]
        lexemes = elements[1].split(',')
        verb_lexemes.update({lexeme: verb for lexeme in lexemes})
    return verb_lexemes


def word_inflections(language):
    inflection_paths = glob.glob(os.path.join(RESOURCES_PATH,
                                              language.iso_code,
                                              "top_*_words_inflected.txt"))
    if len(inflection_paths) == 0:
        return dict()

    inflections = dict()
    with io.open(inflection_paths[0], encoding="utf8") as f:
        lines = [l.strip() for l in f]

    for line in lines:
        elements = line.split(';')
        inflections[elements[0]] = elements[1]
    return inflections


def language_stems(language):
    global _LANGUAGE_STEMS
    if language.iso_code not in _LANGUAGE_STEMS:
        _LANGUAGE_STEMS[language.iso_code] = word_inflections(language)
        _LANGUAGE_STEMS[language.iso_code].update(verbs_lexemes(language))
    return _LANGUAGE_STEMS[language.iso_code]


def stem_sentence(string, language):
    tokens = string.split()
    stemmed_tokens = [stem(token, language) for token in tokens]
    return ' '.join(stemmed_tokens)


def stem(string, language):
    return language_stems(language).get(string, string)
