# coding=utf-8
from __future__ import unicode_literals

import itertools
import re

from num2words import num2words

from snips_nlu.builtin_entities import get_builtin_entities, BuiltInEntity
from snips_nlu.constants import VALUE, MATCH_RANGE
from snips_nlu.languages import Language
from snips_nlu.tokenization import tokenize_light

AND_UTTERANCES = {
    Language.EN: ["and", "&"],
    Language.FR: ["et", "&"],
    Language.ES: ["y", "&"],
    Language.DE: ["und", "&"],
}

AND_REGEXES = {
    language: re.compile(
        r"|".join(r"(?<=\s)%s(?=\s)" % re.escape(u) for u in utterances),
        re.IGNORECASE)
    for language, utterances in AND_UTTERANCES.iteritems()
}


def build_variated_query(string, matches, utterances):
    if not len(matches) == len(utterances):
        raise ValueError("Expected matches and utterances to have the same "
                         "len, found: {} and {}".format(len(matches),
                                                        len(utterances)))
    matches = sorted(matches, key=lambda x: x.start())
    variated_string = ""
    current_ix = 0
    for i, m in enumerate(matches):
        start, end = m.start(), m.end()
        variated_string += string[current_ix:start]
        variated_string += utterances[i]
        current_ix = end
    if current_ix != len(string):
        variated_string += string[current_ix:]
    return variated_string


def and_variations(string, language):
    and_regex = AND_REGEXES.get(language, None)
    variations = set()
    if and_regex is None:
        return variations

    matches = [m for m in and_regex.finditer(string)]
    if len(matches) == 0:
        return variations

    matches = sorted(matches, key=lambda x: x.start())
    and_utterances = AND_UTTERANCES[language]
    combinations = itertools.product(range(len(and_utterances)),
                                     repeat=len(matches))
    for c in combinations:
        utterances = [and_utterances[ix] for ix in c]
        variations.add(build_variated_query(string, matches, utterances))
    return variations


def punctuation_variations(string, language):
    variations = set()
    matches = [m for m in language.punctuation_regex.finditer(string)]
    if len(matches) == 0:
        return variations

    matches = sorted(matches, key=lambda x: x.start())
    punctuation_utterances = [(m.group(0), "") for m in matches]
    combinations = itertools.product(range(2), repeat=len(matches))
    for c in combinations:
        utterances = [punctuation_utterances[i][ix] for i, ix in enumerate(c)]
        variations.add(build_variated_query(string, matches, utterances))
    return variations


def digit_value(number_entity):
    return unicode(number_entity[VALUE][VALUE])


def alphabetic_value(number_entity, language):
    value = number_entity[VALUE][VALUE]
    if isinstance(value, float):  # num2words does not handle floats correctly
        return
    return num2words(value, lang=language.iso_code)


def variate_numbers_in_sentence(string, number_entities, number_utterances,
                                valid_entities_ix):
    valid_entities_ix = set(valid_entities_ix)
    number_entities = sorted(number_entities, key=lambda x: x[MATCH_RANGE])
    variated_string = ""
    current_ix = 0
    for i, ent in enumerate(number_entities):
        if i not in valid_entities_ix:
            continue
        start, end = ent[MATCH_RANGE]
        variated_string += string[current_ix:start]
        variated_string += number_utterances[i]
        current_ix = end
    if current_ix != len(string):
        variated_string += string[current_ix:]
    return variated_string


def numbers_variations(string, language):
    variations = set()
    if not language.supports_num2words:
        return variations

    number_entities = get_builtin_entities(
        string, language, scope=[BuiltInEntity.NUMBER])

    number_entities = [ent for ent in number_entities if
                       not ("latent" in ent[VALUE] and ent[VALUE]["latent"])]
    number_entities = sorted(number_entities, key=lambda x: x["range"])
    if len(number_entities) == 0:
        return variations

    digit_values = [digit_value(e) for e in number_entities]
    alpha_values = [alphabetic_value(e, language) for e in number_entities]

    valid_values_ix = [i for i, (d, a) in
                       enumerate(itertools.izip(digit_values, alpha_values))
                       if a is not None]

    values = zip(digit_values, alpha_values)

    combinations = itertools.product(xrange(2), repeat=len(number_entities))
    for c in combinations:
        number_utterances = [values[i][ix] for i, ix in enumerate(c)]
        variations.add(variate_numbers_in_sentence(
            string, number_entities, number_utterances, valid_values_ix))
    return variations


def flatten(results):
    return set(i for r in results for i in r)


def get_string_variations(string, language):
    variations = {string}
    variations.update(flatten(map(lambda x: and_variations(x, language),
                                  variations)))
    variations.update(flatten(
        map(lambda x: punctuation_variations(x, language), variations)))
    variations.update(flatten(map(lambda x: numbers_variations(x, language),
                                  variations)))

    variations = set(language.default_sep.join(tokenize_light(v, language))
                     for v in variations)
    return variations
