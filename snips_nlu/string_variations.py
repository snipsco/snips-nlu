# coding=utf-8
from __future__ import unicode_literals

import itertools
import re
from builtins import range, str, zip

from future.utils import iteritems
from num2words import num2words
from snips_nlu_utils import normalize

from snips_nlu.constants import (
    END, ENTITY, LANGUAGE_DE, LANGUAGE_EN, LANGUAGE_ES, LANGUAGE_FR,
    RES_MATCH_RANGE, SNIPS_NUMBER, START, VALUE)
from snips_nlu.languages import (
    get_default_sep, get_punctuation_regex, supports_num2words)
from snips_nlu.preprocessing import tokenize_light

AND_UTTERANCES = {
    LANGUAGE_EN: ["and", "&"],
    LANGUAGE_FR: ["et", "&"],
    LANGUAGE_ES: ["y", "&"],
    LANGUAGE_DE: ["und", "&"],
}

AND_REGEXES = {
    language: re.compile(
        r"|".join(r"(?<=\s)%s(?=\s)" % re.escape(u) for u in utterances),
        re.IGNORECASE)
    for language, utterances in iteritems(AND_UTTERANCES)
}

MAX_ENTITY_VARIATIONS = 10


def build_variated_query(string, ranges_and_utterances):
    variated_string = ""
    current_ix = 0
    for rng, u in ranges_and_utterances:
        start = rng[START]
        end = rng[END]
        variated_string += string[current_ix:start]
        variated_string += u
        current_ix = end
    variated_string += string[current_ix:]
    return variated_string


def and_variations(string, language):
    and_regex = AND_REGEXES.get(language, None)
    if and_regex is None:
        return set()

    matches = [m for m in and_regex.finditer(string)]
    if not matches:
        return set()

    matches = sorted(matches, key=lambda x: x.start())
    and_utterances = AND_UTTERANCES[language]
    values = [({START: m.start(), END: m.end()}, and_utterances)
              for m in matches]

    n_values = len(values)
    n_and_utterances = len(and_utterances)
    if n_and_utterances ** n_values > MAX_ENTITY_VARIATIONS:
        return set()

    combinations = itertools.product(range(n_and_utterances), repeat=n_values)
    variations = set()
    for c in combinations:
        ranges_and_utterances = [(values[i][0], values[i][1][ix])
                                 for i, ix in enumerate(c)]
        variations.add(build_variated_query(string, ranges_and_utterances))
    return variations


def punctuation_variations(string, language):
    matches = [m for m in get_punctuation_regex(language).finditer(string)]
    if not matches:
        return set()

    matches = sorted(matches, key=lambda x: x.start())
    values = [({START: m.start(), END: m.end()}, (m.group(0), ""))
              for m in matches]

    n_values = len(values)
    if 2 ** n_values > MAX_ENTITY_VARIATIONS:
        return set()

    combinations = itertools.product(range(2), repeat=n_values)
    variations = set()
    for c in combinations:
        ranges_and_utterances = [(values[i][0], values[i][1][ix])
                                 for i, ix in enumerate(c)]
        variations.add(build_variated_query(string, ranges_and_utterances))
    return variations


def digit_value(number_entity):
    value = number_entity[ENTITY][VALUE]
    if value == int(value):
        # Convert 24.0 into "24" instead of "24.0"
        value = int(value)
    return str(value)


def alphabetic_value(number_entity, language):
    value = number_entity[ENTITY][VALUE]
    if value != int(value):  # num2words does not handle floats correctly
        return None
    return num2words(int(value), lang=language)


def numbers_variations(string, language, builtin_entity_parser):
    if not supports_num2words(language):
        return set()

    number_entities = builtin_entity_parser.parse(
        string, scope=[SNIPS_NUMBER], use_cache=True)

    number_entities = sorted(number_entities,
                             key=lambda x: x[RES_MATCH_RANGE][START])
    if not number_entities:
        return set()

    digit_values = [digit_value(e) for e in number_entities]
    alpha_values = [alphabetic_value(e, language) for e in number_entities]

    values = [(n[RES_MATCH_RANGE], (d, a)) for (n, d, a) in
              zip(number_entities, digit_values, alpha_values)
              if a is not None]

    n_values = len(values)
    if 2 ** n_values > MAX_ENTITY_VARIATIONS:
        return set()

    combinations = itertools.product(range(2), repeat=n_values)
    variations = set()
    for c in combinations:
        ranges_and_utterances = [(values[i][0], values[i][1][ix])
                                 for i, ix in enumerate(c)]
        variations.add(build_variated_query(string, ranges_and_utterances))
    return variations


def case_variations(string):
    return {string.lower(), string.title()}


def normalization_variations(string):
    return {normalize(string)}


def flatten(results):
    return set(i for r in results for i in r)


def get_string_variations(string, language, builtin_entity_parser):
    variations = {string}
    variations.update(flatten(case_variations(v) for v in variations))
    variations.update(flatten(normalization_variations(v) for v in variations))
    # We re-generate case variations as normalization can produce new
    # variations
    variations.update(flatten(case_variations(v) for v in variations))
    variations.update(flatten(and_variations(v, language) for v in variations))
    variations.update(
        flatten(punctuation_variations(v, language) for v in variations))
    variations.update(
        flatten(numbers_variations(v, language, builtin_entity_parser)
                for v in variations))
    # Add single space variations
    single_space_variations = set(" ".join(v.split()) for v in variations)
    variations.update(single_space_variations)
    # Add tokenized variations
    tokenized_variations = set(
        get_default_sep(language).join(tokenize_light(v, language)) for v in
        variations)
    variations.update(tokenized_variations)
    return variations
