from __future__ import absolute_import, print_function, unicode_literals

from abc import ABCMeta, abstractmethod
from builtins import object
from io import IOBase

import yaml
from future.utils import with_metaclass

from snips_nlu.constants import DATA, ENTITY, SLOT_NAME, TEXT, UTTERANCES
from snips_nlu.exceptions import IntentFormatError


class Intent(object):
    """Intent data of a :class:`.Dataset`

    Attributes:
        intent_name (str): name of the intent
        utterances (list of :class:`.IntentUtterance`): annotated intent
            utterances
        slot_mapping (dict): mapping between slot names and entities
    """

    def __init__(self, intent_name, utterances, slot_mapping=None):
        if slot_mapping is None:
            slot_mapping = dict()
        self.intent_name = intent_name
        self.utterances = utterances
        self.slot_mapping = slot_mapping
        self._complete_slot_name_mapping()
        self._ensure_entity_names()

    @classmethod
    def from_yaml(cls, yaml_dict):
        """Build an :class:`.Intent` from its YAML definition object

        Args:
            yaml_dict (dict or :class:`.IOBase`): object containing the YAML
                definition of the intent. It can be either a stream, or the
                corresponding python dict.

        Examples:
            An intent can be defined with a YAML document following the schema
            illustrated in the example below:

            >>> import io
            >>> from snips_nlu.common.utils import json_string
            >>> intent_yaml = io.StringIO('''
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
            ...   - I need a flight leaving to [destination](Berlin)''')
            >>> intent = Intent.from_yaml(intent_yaml)
            >>> print(json_string(intent.json, indent=4, sort_keys=True))
            {
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

        Raises:
            IntentFormatError: When the YAML dict does not correspond to the
                :ref:`expected intent format <yaml_intent_format>`
        """
        if isinstance(yaml_dict, IOBase):
            yaml_dict = yaml.safe_load(yaml_dict)

        object_type = yaml_dict.get("type")
        if object_type and object_type != "intent":
            raise IntentFormatError("Wrong type: '%s'" % object_type)
        intent_name = yaml_dict.get("name")
        if not intent_name:
            raise IntentFormatError("Missing 'name' attribute")
        slot_mapping = dict()
        for slot in yaml_dict.get("slots", []):
            slot_mapping[slot["name"]] = slot["entity"]
        utterances = [IntentUtterance.parse(u.strip())
                      for u in yaml_dict["utterances"] if u.strip()]
        if not utterances:
            raise IntentFormatError(
                "Intent must contain at least one utterance")
        return cls(intent_name, utterances, slot_mapping)

    def _complete_slot_name_mapping(self):
        for utterance in self.utterances:
            for chunk in utterance.slot_chunks:
                if chunk.entity and chunk.slot_name not in self.slot_mapping:
                    self.slot_mapping[chunk.slot_name] = chunk.entity
        return self

    def _ensure_entity_names(self):
        for utterance in self.utterances:
            for chunk in utterance.slot_chunks:
                if chunk.entity:
                    continue
                chunk.entity = self.slot_mapping.get(
                    chunk.slot_name, chunk.slot_name)
        return self

    @property
    def json(self):
        """Intent data in json format"""
        return {
            UTTERANCES: [
                {DATA: [chunk.json for chunk in utterance.chunks]}
                for utterance in self.utterances
            ]
        }

    @property
    def entities_names(self):
        return set(chunk.entity for u in self.utterances
                   for chunk in u.chunks if isinstance(chunk, SlotChunk))


class IntentUtterance(object):
    def __init__(self, chunks):
        self.chunks = chunks

    @property
    def text(self):
        return "".join((chunk.text for chunk in self.chunks))

    @property
    def slot_chunks(self):
        return (chunk for chunk in self.chunks if isinstance(chunk, SlotChunk))

    @classmethod
    def parse(cls, string):
        """Parses an utterance

        Args:
            string (str): an utterance in the class:`.Utterance` format

        Examples:

            >>> from snips_nlu.dataset.intent import IntentUtterance
            >>> u = IntentUtterance.\
                parse("president of [country:default](France)")
            >>> u.text
            'president of France'
            >>> len(u.chunks)
            2
            >>> u.chunks[0].text
            'president of '
            >>> u.chunks[1].slot_name
            'country'
            >>> u.chunks[1].entity
            'default'
        """
        sm = SM(string)
        capture_text(sm)
        return cls(sm.chunks)


class Chunk(with_metaclass(ABCMeta, object)):
    def __init__(self, text):
        self.text = text

    @abstractmethod
    def json(self):
        pass


class SlotChunk(Chunk):
    def __init__(self, slot_name, entity, text):
        super(SlotChunk, self).__init__(text)
        self.slot_name = slot_name
        self.entity = entity

    @property
    def json(self):
        return {
            TEXT: self.text,
            SLOT_NAME: self.slot_name,
            ENTITY: self.entity,
        }


class TextChunk(Chunk):
    @property
    def json(self):
        return {
            TEXT: self.text
        }


class SM(object):
    """State Machine for parsing"""

    def __init__(self, input):
        self.input = input
        self.chunks = []
        self.current = 0

    @property
    def end_of_input(self):
        return self.current >= len(self.input)

    def add_slot(self, name, entity=None):
        """Adds a named slot

        Args:
            name (str): slot name
            entity (str): entity name
        """
        chunk = SlotChunk(slot_name=name, entity=entity, text=None)
        self.chunks.append(chunk)

    def add_text(self, text):
        """Adds a simple text chunk using the current position"""
        chunk = TextChunk(text=text)
        self.chunks.append(chunk)

    def add_tagged(self, text):
        """Adds text to the last slot"""
        if not self.chunks:
            raise AssertionError("Cannot add tagged text because chunks list "
                                 "is empty")
        self.chunks[-1].text = text

    def find(self, s):
        return self.input.find(s, self.current)

    def move(self, pos):
        """Moves the cursor of the state to position after given

        Args:
            pos (int): position to place the cursor just after
        """
        self.current = pos + 1

    def peek(self):
        if self.end_of_input:
            return None
        return self[0]

    def read(self):
        c = self[0]
        self.current += 1
        return c

    def __getitem__(self, key):
        current = self.current
        if isinstance(key, int):
            return self.input[current + key]
        elif isinstance(key, slice):
            start = current + key.start if key.start else current
            return self.input[slice(start, key.stop, key.step)]
        else:
            raise TypeError("Bad key type: %s" % type(key))


def capture_text(state):
    next_pos = state.find('[')
    sub = state[:] if next_pos < 0 else state[:next_pos]
    if sub:
        state.add_text(sub)
    if next_pos >= 0:
        state.move(next_pos)
        capture_slot(state)


def capture_slot(state):
    next_colon_pos = state.find(':')
    next_square_bracket_pos = state.find(']')
    if next_square_bracket_pos < 0:
        raise IntentFormatError("Missing ending ']' in annotated utterance")
    if next_colon_pos < 0 or next_square_bracket_pos < next_colon_pos:
        slot_name = state[:next_square_bracket_pos]
        state.move(next_square_bracket_pos)
        state.add_slot(slot_name)
    else:
        slot_name = state[:next_colon_pos]
        state.move(next_colon_pos)
        entity = state[:next_square_bracket_pos]
        state.move(next_square_bracket_pos)
        state.add_slot(slot_name, entity)
    if state.peek() == '(':
        state.read()
        capture_tagged(state)
    else:
        capture_text(state)


def capture_tagged(state):
    next_pos = state.find(')')
    if next_pos < 1:
        raise IntentFormatError("Missing ending ')' in annotated utterance")
    else:
        tagged_text = state[:next_pos]
        state.add_tagged(tagged_text)
        state.move(next_pos)
        capture_text(state)
