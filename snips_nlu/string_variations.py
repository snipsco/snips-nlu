# coding=utf-8
from __future__ import unicode_literals

import itertools
import re

from builtins import range
from builtins import str
from builtins import zip
from num2words import num2words
from six import iteritems

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
    for language, utterances in iteritems(AND_UTTERANCES)
}


def build_variated_query(string, ranges_and_utterances):
    variated_string = ""
    current_ix = 0
    for m, u in ranges_and_utterances:
        start, end = m
        variated_string += string[current_ix:start]
        variated_string += u
        current_ix = end
    variated_string += string[current_ix:]
    return variated_string


def and_variations(string, language):
    and_regex = AND_REGEXES.get(language, None)
    variations = set()
    if and_regex is None:
        return variations

    matches = [m for m in and_regex.finditer(string)]
    if not matches:
        return variations

    matches = sorted(matches, key=lambda x: x.start())
    values = [((m.start(), m.end()), AND_UTTERANCES[language])
              for m in matches]
    combinations = itertools.product(
        range(len(AND_UTTERANCES[language])), repeat=len(values))
    for c in combinations:
        ranges_and_utterances = [(values[i][0], values[i][1][ix])
                                 for i, ix in enumerate(c)]
        variations.add(build_variated_query(string, ranges_and_utterances))
    return variations


def punctuation_variations(string, language):
    variations = set()
    matches = [m for m in language.punctuation_regex.finditer(string)]
    if not matches:
        return variations

    matches = sorted(matches, key=lambda x: x.start())
    values = [((m.start(), m.end()), (m.group(0), "")) for m in matches]

    combinations = itertools.product(range(2), repeat=len(matches))
    for c in combinations:
        ranges_and_utterances = [(values[i][0], values[i][1][ix])
                                 for i, ix in enumerate(c)]
        variations.add(build_variated_query(string, ranges_and_utterances))
    return variations


def digit_value(number_entity):
    return str(number_entity[VALUE][VALUE])


def alphabetic_value(number_entity, language):
    value = number_entity[VALUE][VALUE]
    if isinstance(value, float):  # num2words does not handle floats correctly
        return
    return num2words(value, lang=language.iso_code)


def numbers_variations(string, language):
    variations = set()
    if not language.supports_num2words:
        return variations

    number_entities = get_builtin_entities(
        string, language, scope=[BuiltInEntity.NUMBER])

    number_entities = [ent for ent in number_entities if
                       not ("latent" in ent[VALUE] and ent[VALUE]["latent"])]
    number_entities = sorted(number_entities, key=lambda x: x["range"])
    if not number_entities:
        return variations

    digit_values = [digit_value(e) for e in number_entities]
    alpha_values = [alphabetic_value(e, language) for e in number_entities]

    values = [(n[MATCH_RANGE], (d, a)) for (n, d, a) in
              zip(number_entities, digit_values, alpha_values)
              if a is not None]

    combinations = itertools.product(range(2), repeat=len(values))
    for c in combinations:
        ranges_and_utterances = [(values[i][0], values[i][1][ix])
                                 for i, ix in enumerate(c)]
        variations.add(build_variated_query(string, ranges_and_utterances))
    return variations


def flatten(results):
    return set(i for r in results for i in r)


def get_string_variations(string, language):
    variations = {string}
    variations.update(flatten(and_variations(v, language) for v in variations))
    variations.update(
        flatten(punctuation_variations(v, language) for v in variations))
    variations.update(
        flatten(numbers_variations(v, language) for v in variations))
    variations = set(language.default_sep.join(tokenize_light(v, language))
                     for v in variations)
    return variations
