# coding=utf-8
from __future__ import print_function, unicode_literals

import io
from itertools import cycle

import yaml
from snips_nlu_parsers import get_builtin_entity_examples

from snips_nlu.common.utils import unicode_string
from snips_nlu.dataset.entity import Entity
from snips_nlu.dataset.intent import Intent
from snips_nlu.exceptions import DatasetFormatError


class Dataset(object):
    """Dataset used in the main NLU training API

    Consists of intents and entities data. This object can be built either from
    text files (:meth:`.Dataset.from_files`) or from YAML files
    (:meth:`.Dataset.from_yaml_files`).

    Attributes:
        language (str): language of the intents
        intents (list of :class:`.Intent`): intents data
        entities (list of :class:`.Entity`): entities data
    """

    def __init__(self, language, intents, entities):
        self.language = language
        self.intents = intents
        self.entities = entities
        self._add_missing_entities()
        self._ensure_entity_values()

    @classmethod
    def from_yaml_files(cls, language, filenames):
        """Creates a :class:`.Dataset` from a language and a list of YAML files
        or streams containing intents and entities data

        Each file need not correspond to a single entity nor intent. They can
        consist in several entities and intents merged together in a single
        file.

        Args:
            language (str): language of the dataset (ISO639-1)
            filenames (iterable): filenames or stream objects corresponding to
                intents and entities data.

        Example:

            A dataset can be defined with a YAML document following the schema
            illustrated in the example below:

            >>> import io
            >>> from snips_nlu.common.utils import json_string
            >>> dataset_yaml = io.StringIO('''
            ... # searchFlight Intent
            ... ---
            ... type: intent
            ... name: searchFlight
            ... slots:
            ...   - name: origin
            ...     entity: city
            ...   - name: destination
            ...     entity: city
            ...   - name: date
            ...     entity: snips/datetime
            ... utterances:
            ...   - find me a flight from [origin](Oslo) to [destination](Lima)
            ...   - I need a flight leaving to [destination](Berlin)
            ...
            ... # City Entity
            ... ---
            ... type: entity
            ... name: city
            ... values:
            ...   - london
            ...   - [paris, city of lights]''')
            >>> dataset = Dataset.from_yaml_files("en", [dataset_yaml])
            >>> print(json_string(dataset.json, indent=4, sort_keys=True))
            {
                "entities": {
                    "city": {
                        "automatically_extensible": true,
                        "data": [
                            {
                                "synonyms": [],
                                "value": "london"
                            },
                            {
                                "synonyms": [
                                    "city of lights"
                                ],
                                "value": "paris"
                            }
                        ],
                        "matching_strictness": 1.0,
                        "use_synonyms": true
                    }
                },
                "intents": {
                    "searchFlight": {
                        "utterances": [
                            {
                                "data": [
                                    {
                                        "text": "find me a flight from "
                                    },
                                    {
                                        "entity": "city",
                                        "slot_name": "origin",
                                        "text": "Oslo"
                                    },
                                    {
                                        "text": " to "
                                    },
                                    {
                                        "entity": "city",
                                        "slot_name": "destination",
                                        "text": "Lima"
                                    }
                                ]
                            },
                            {
                                "data": [
                                    {
                                        "text": "I need a flight leaving to "
                                    },
                                    {
                                        "entity": "city",
                                        "slot_name": "destination",
                                        "text": "Berlin"
                                    }
                                ]
                            }
                        ]
                    }
                },
                "language": "en"
            }

        Raises:
            DatasetFormatError: When one of the documents present in the YAML
                files has a wrong 'type' attribute, which is not 'entity' nor
                'intent'
            IntentFormatError: When the YAML document of an intent does not
                correspond to the
                :ref:`expected intent format <yaml_intent_format>`
            EntityFormatError: When the YAML document of an entity does not
                correspond to the
                :ref:`expected entity format <yaml_entity_format>`
        """
        language = unicode_string(language)
        entities = []
        intents = []
        for filename in filenames:
            if isinstance(filename, io.IOBase):
                intents_, entities_ = cls._load_dataset_parts(
                    filename, "stream object")
            else:
                with io.open(filename, encoding="utf8") as f:
                    intents_, entities_ = cls._load_dataset_parts(f, filename)
            intents += intents_
            entities += entities_
        return cls(language, intents, entities)

    @classmethod
    def _load_dataset_parts(cls, stream, stream_description):
        intents = []
        entities = []
        for doc in yaml.safe_load_all(stream):
            doc_type = doc.get("type")
            if doc_type == "entity":
                entities.append(Entity.from_yaml(doc))
            elif doc_type == "intent":
                intents.append(Intent.from_yaml(doc))
            else:
                raise DatasetFormatError(
                    "Invalid 'type' value in YAML file '%s': '%s'"
                    % (stream_description, doc_type))
        return intents, entities

    def _add_missing_entities(self):
        entity_names = set(e.name for e in self.entities)

        # Add entities appearing only in the intents utterances
        for intent in self.intents:
            for entity_name in intent.entities_names:
                if entity_name not in entity_names:
                    entity_names.add(entity_name)
                    self.entities.append(Entity(name=entity_name))

    def _ensure_entity_values(self):
        entities_values = {entity.name: self._get_entity_values(entity)
                           for entity in self.entities}
        for intent in self.intents:
            for utterance in intent.utterances:
                for chunk in utterance.slot_chunks:
                    if chunk.text is not None:
                        continue
                    try:
                        chunk.text = next(entities_values[chunk.entity])
                    except StopIteration:
                        raise DatasetFormatError(
                            "At least one entity value must be provided for "
                            "entity '%s'" % chunk.entity)
        return self

    def _get_entity_values(self, entity):
        if entity.is_builtin:
            return cycle(get_builtin_entity_examples(
                entity.name, self.language))
        values = [v for utterance in entity.utterances
                  for v in utterance.variations]
        values_set = set(values)
        for intent in self.intents:
            for utterance in intent.utterances:
                for chunk in utterance.slot_chunks:
                    if not chunk.text or chunk.entity != entity.name:
                        continue
                    if chunk.text not in values_set:
                        values_set.add(chunk.text)
                        values.append(chunk.text)
        return cycle(values)

    @property
    def json(self):
        """Dataset data in json format"""
        intents = {intent_data.intent_name: intent_data.json
                   for intent_data in self.intents}
        entities = {entity.name: entity.json for entity in self.entities}
        return dict(language=self.language, intents=intents, entities=entities)
