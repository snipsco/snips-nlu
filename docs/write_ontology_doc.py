from pathlib import Path

from snips_nlu_parsers.builtin_entities import (
    get_complete_entity_ontology, get_all_gazetteer_entities)

ONTOLOGY = get_complete_entity_ontology()
ALL_GAZETTEER_ENTITIES = get_all_gazetteer_entities()
LANGUAGE_DOC_PATH = Path(__file__).parent / "source" / "languages.rst"
ENTITIES_DOC_PATH = Path(__file__).parent / "source" / "builtin_entities.rst"

GRAMMAR_ENTITY = "Grammar Entity"
GAZETTEER_ENTITY = "Gazetteer Entity"

LANGUAGES_DOC_HEADER = """.. _languages:

Supported languages
===================

Snips NLU supports various languages, that are specified in the dataset in the
``"language"`` attribute. Here is the list of supported language along with
their isocode:
"""

LANGUAGES_DOC_FOOTER = """

Support for additional languages will come in the future, stay tuned :)
"""

ENTITIES_DOC_HEADER = """.. _builtin_entities:

Supported builtin entities
==========================

:ref:`Builtin entities <builtin_entity_resolution>` are entities that have
a built-in support in Snips NLU. These entities are associated to specific
builtin entity parsers which provide an extra resolution step. Typically,
dates written in natural language (``"in three days"``) are resolved into ISO
formatted dates (``"2019-08-12 00:00:00 +02:00"``).

Here is the list of supported builtin entities:

"""

ENTITIES_DOC_MIDDLE = """

The entity identifier (second column above) is what is used in the dataset to
reference a builtin entity.

Grammar Entity
--------------

Grammar entities, in the context of Snips NLU, correspond to entities which 
contain significant `compositionality`_. The semantic meaning of such an 
entity is determined by the meanings of its constituent expressions and the 
rules used to combine them. Modern semantic parsers for these entities are 
often based on defining a formal grammar. In the case of Snips NLU, the parser 
used to handle these entities is `Rustling`_, a Rust adaptation of Facebook's 
`duckling`_.

Gazetteer Entity
----------------

Gazetteer entities correspond to all the builtin entities which do not contain 
any semantic structure, as opposed to the grammar entities. For such 
entities, a `gazetteer entity parser`_ is used to perform the parsing.

Results Examples
----------------

The following sections provide examples for each builtin entity. 

"""

ENTITIES_DOC_FOOTER = """
.. _compositionality: https://en.wikipedia.org/wiki/Principle_of_compositionality
.. _Rustling: https://github.com/snipsco/rustling-ontology
.. _duckling: https://github.com/facebook/duckling
.. _gazetteer entity parser: https://github.com/snipsco/gazetteer-entity-parser
"""

LANGUAGES_TABLE_CELL_LENGTH = 12
ENTITIES_TABLE_CELL_LENGTH = 50


def write_supported_languages(path):
    languages = sorted([lang_ontology["language"]
                        for lang_ontology in ONTOLOGY])
    table = _build_supported_languages_table(languages)
    content = LANGUAGES_DOC_HEADER + table + LANGUAGES_DOC_FOOTER
    with path.open(mode="w") as f:
        f.write(content)


def write_supported_builtin_entities(path):
    table = _build_supported_entities_table(ONTOLOGY)
    results_examples = _build_results_examples(ONTOLOGY)
    content = ENTITIES_DOC_HEADER + table + ENTITIES_DOC_MIDDLE + \
              results_examples + ENTITIES_DOC_FOOTER
    with path.open(mode="w") as f:
        f.write(content)


def _build_supported_languages_table(languages):
    table = _build_table_cells(["ISO code"], LANGUAGES_TABLE_CELL_LENGTH, "=",
                               "-")
    for language in languages:
        table += _build_table_cells([language], LANGUAGES_TABLE_CELL_LENGTH,
                                    "-")
    return table


def _build_supported_entities_table(ontology):
    en_ontology = None
    for lang_ontology in ontology:
        if lang_ontology["language"] == "en":
            en_ontology = lang_ontology
            break
    table = _build_table_cells(
        ["Entity", "Identifier", "Category", "Supported Languages"],
        ENTITIES_TABLE_CELL_LENGTH, "=", "-")
    for entity in en_ontology["entities"]:
        table += _build_table_cells(
            ["`%s`_" % entity["name"], entity["label"],
             "`%s`_" % _category(entity["label"]),
             ", ".join(entity["supportedLanguages"])],
            ENTITIES_TABLE_CELL_LENGTH, "-")
    return table


def _build_results_examples(ontology):
    content = ""
    en_ontology = None
    for lang_ontology in ontology:
        if lang_ontology["language"] == "en":
            en_ontology = lang_ontology
            break
    for entity in en_ontology["entities"]:
        name = entity["name"]
        title = "\n".join([len(name) * "-", name, len(name) * "-"])
        input_examples = """
Input examples:

.. code-block:: json

   [
     %s
   ]
""" % (",\n     ".join(["\"%s\"" % ex for ex in entity["examples"]]))
        output_examples = """
Output examples:

.. code-block:: json

   %s

""" % entity["resultDescription"].replace("\n", "\n   ")
        content += "\n".join([title, input_examples, output_examples])
    return content


def _build_table_cells(contents, cell_length, bottom_sep_char,
                       top_sep_char=None):
    cells = []
    for i, content in enumerate(contents):
        right_bar = ""
        right_plus = ""
        if i == len(contents) - 1:
            right_bar = "|"
            right_plus = "+"
        blank_suffix_length = cell_length - len(content) - 1
        blank_suffix = blank_suffix_length * " "
        cell_prefix = ""
        if top_sep_char is not None:
            top_line_sep = cell_length * top_sep_char
            cell_prefix = "+%s%s\n" % (top_line_sep, right_plus)
        bottom_line_sep = cell_length * bottom_sep_char
        cell = """
%s| %s%s%s
+%s%s""" % (cell_prefix, content, blank_suffix, right_bar, bottom_line_sep,
            right_plus)
        cells.append(cell)
    cell_lines = zip(*(c.split("\n") for c in cells))
    cell_lines = ["".join(line) for line in cell_lines]
    cell = "\n".join(cell_lines)
    return cell


def _category(entity_identifier):
    if entity_identifier in ALL_GAZETTEER_ENTITIES:
        return GAZETTEER_ENTITY
    return GRAMMAR_ENTITY


if __name__ == "__main__":
    write_supported_languages(LANGUAGE_DOC_PATH)
    write_supported_builtin_entities(ENTITIES_DOC_PATH)
