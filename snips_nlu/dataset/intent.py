from __future__ import absolute_import, print_function

from abc import ABCMeta, abstractmethod
from builtins import object
from pathlib import Path

from future.utils import with_metaclass

from snips_nlu.constants import DATA, ENTITY, SLOT_NAME, TEXT, UTTERANCES


class IntentFormatError(TypeError):
    pass


INTENT_FORMATTING_ERROR = IntentFormatError(
    "Intent file is not properly formatted")


class Intent(object):
    """Dataset of an intent

    Can parse utterances from a text file or an iterator.

    An example of utterance is:

        "the [role:role](president) of [country:country](France)"

    a Tag is in this format:

        [slot:entity_name](text_to_tag)

    Attributes:
        intent_name (str): name of the intent
        utterances (list of :class:`.IntentUtterance`): intent utterances
    """

    def __init__(self, intent_name):
        self.intent_name = intent_name
        self.utterances = []
        self.slot_mapping = dict()

    @classmethod
    def from_file(cls, filepath):
        filepath = Path(filepath)
        stem = filepath.stem
        if not stem.startswith("intent_"):
            raise IntentFormatError(
                "Intent filename should start with 'intent_' but found: %s"
                % stem)
        intent_name = stem[7:]
        if not intent_name:
            raise IntentFormatError("Intent name must not be empty")
        with filepath.open(encoding="utf-8") as f:
            lines = iter(l.strip() for l in f if l.strip())
            return cls.from_iter(intent_name, lines)

    @classmethod
    def from_iter(cls, intent_name, samples_iter):
        """Generates a dataset from an iterator of samples"""
        dataset = cls(intent_name)
        for sample in samples_iter:
            utterance = IntentUtterance.parse(sample)
            dataset.add(utterance)
        return dataset

    def add(self, utterance):
        """Adds an :class:`.IntentUtterance` to the dataset"""
        for chunk in utterance.slot_chunks:
            if chunk.name not in self.slot_mapping:
                self.slot_mapping[chunk.name] = chunk.entity
        self.utterances.append(utterance)

    @property
    def json(self):
        """Intent dataset in json format"""
        return {
            UTTERANCES: [
                {DATA: [chunk.json for chunk in utterance.chunks]}
                for utterance in self.utterances
            ]
        }

    @property
    def entities_names(self):
        """Set of entity names present in the intent dataset"""
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
            >>> u.chunks[1].name
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
        self.name = slot_name
        self.entity = entity

    @property
    def json(self):
        return {
            TEXT: self.text,
            SLOT_NAME: self.name,
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

    def add_slot(self, name, entity):
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
    if sub.strip():
        state.add_text(sub)
    if next_pos >= 0:
        state.move(next_pos)
        capture_slot(state)


def capture_slot(state):
    next_pos = state.find(':')
    if next_pos < 0:
        raise INTENT_FORMATTING_ERROR
    else:
        slot_name = state[:next_pos]
        state.move(next_pos)
        next_pos = state.find(']')
        if next_pos < 0:
            raise INTENT_FORMATTING_ERROR
        entity = state[:next_pos]
        state.move(next_pos)
        state.add_slot(slot_name, entity)
        if state.read() != '(':
            raise INTENT_FORMATTING_ERROR
        capture_tagged(state)


def capture_tagged(state):
    next_pos = state.find(')')
    if next_pos < 1:
        raise INTENT_FORMATTING_ERROR
    else:
        tagged_text = state[:next_pos]
        state.add_tagged(tagged_text)
        state.move(next_pos)
        capture_text(state)
