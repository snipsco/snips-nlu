from __future__ import print_function, absolute_import

import io
import itertools
import os

from builtins import object

from snips_nlu_dataset.builtin_entities import BuiltinEntity
from snips_nlu_dataset.custom_entities import CustomEntity, EntityUtterance
from snips_nlu.builtin_entities import is_builtin_entity


class IntentDataset(object):
    """Dataset of an intent

    Can parse utterances from a text file or an iterator.

    An example of utterance is:

        "the [role:role](president) of [country:country](France)"

    a Tag is in this format:

        [slot:entity_name](text_to_tag)

    Attributes:
        :class:IntentDataset.utterances: a list of :class:Utterance
        :class:IntentDataset.json: The dataset in json format
        :class:IntentDataset.intent_name: name of the intent
        :class:IntentDataset.language: language of the dataset
    """

    def __init__(self, intent_name, language):
        self.intent_name = intent_name
        self.language = language
        self.slots_by_name = dict()
        self.json_utterances = []
        self.utterances = []

    @classmethod
    def from_file(cls, language, file_name):
        intent_name = ".".join(os.path.basename(file_name).split('.')[:-1])
        with io.open(file_name) as f:
            lines = iter(l.strip() for l in f if l.strip() != '')
            return cls.from_iter(intent_name, language, lines)

    @classmethod
    def from_iter(cls, intent_name, language, samples_iter):
        """Generates a dataset from an iterator of samples

        :param intent_name: string
        :param language: string
        :param samples_iter: an iterator of string samples
        :return: :class:Dataset
        """
        dataset = cls(intent_name, language)
        for sample in samples_iter:
            u = Utterance.parse(sample)
            dataset.add(u)
        return dataset

    def add(self, utterance):
        """Adds an utterance to the dataset

        :param utterance: :class:Utterance
        """
        self.utterances.append(utterance)
        data = []
        for slot in utterance.slots:
            data.append(self.mk_slot(slot))
        self.json_utterances.append(dict(data=data))

    @property
    def json(self):
        """Dataset in json format"""
        return dict(language=self.language,
                    utterances=self.json_utterances,
                    entities=self.entities)

    @property
    def queries(self):
        """:return: An iter of all the example queries"""
        return (u.input for u in self.utterances)

    @property
    def entities(self):
        """retun all entities in json format for datasets"""
        ents = dict()
        for s in self.slots:
            if s.entity not in ents:
                ents[s.entity] = self.mk_entity(s)
            elif not is_builtin_entity(s.entity):
                ents[s.entity].utterances.append(EntityUtterance(s.text))
        return ents

    @classmethod
    def mk_entity(cls, slot, automatically_extensible=True, use_synonyms=True):
        if is_builtin_entity(slot.entity):
            return BuiltinEntity(slot.entity)
        return CustomEntity([EntityUtterance(slot.text)],
                            automatically_extensible, use_synonyms)

    @property
    def json_slots(self):
        """:return: slots but in json format"""
        return (self.intent_slot(s) for s in self.slots)

    @property
    def slots(self):
        """:return: an iter of the unique slots in the dataset"""
        all_slots = [
            s for u in self.utterances
            for s in u.slots
            if isinstance(s, Slot)
        ]
        all_slots.sort(key=lambda s: s.name)
        groups = itertools.groupby(all_slots, lambda s: s.entity)
        return (s for r, g in groups for s in g)

    @property
    def annotated(self):
        return (u.annotated for u in self.utterances)

    @staticmethod
    def intent_slot(slot):
        """json format expected in intent data"""
        return dict(
            name=slot.name,
            entity=slot.entity_name,
            entity_id=slot.entity
        )

    @staticmethod
    def mk_slot(slot):
        if isinstance(slot, Slot):
            return dict(
                slot_name=slot.name,
                entity=slot.entity,
                text=slot.text,
            )

        return dict(
            text=slot.text,
        )


class Utterance(object):
    def __init__(self, input, slots):
        self.input = input
        self.slots = slots

    @property
    def annotated(self):
        """Annotate with *

        :return: the sentence annotated just with stars
        >>> p = "the [role:snips/def](president) of [c:snips/def](France)"
        >>> u = Utterance.parse(p)
        >>> u.annotated
        u'the *president* of *France*'
        """
        binput = bytearray(self.input, 'utf-8')
        acc = 0
        star = ord('*')
        for slot in self.slots:
            if isinstance(slot, Slot):
                binput.insert(slot.range.start + acc, star)
                binput.insert(slot.range.end + acc + 1, star)
                acc += 2
        return binput.decode('utf-8')

    @staticmethod
    def stripped(input, slots):
        acc = 0
        s = ''
        new_slots = []
        for slot in slots:
            start = slot.range.start
            end = slot.range.end
            s += input[start:end]
            if isinstance(slot, Slot):
                acc += slot.tag_range.size
                range = Range(start - acc, end - acc)
                new_slot = Slot(slot.name, slot.entity, range, slot.text,
                                slot.tag_range)
                new_slots.append(new_slot)
                acc += 1
            else:
                range = Range(start - acc, end - acc)
                new_slots.append(Text(slot.text, range))
        return s, new_slots

    @staticmethod
    def parse(string):
        """Parses an utterance.

        :param string: an utterance in the Utterance format

        >>> u = Utterance.parse("president of [country:default](France)")
        >>> len(u.slots)
        2
        >>> u.slots[0].text
        'president of '
        >>> u.slots[0].range.start
        0
        >>> u.slots[0].range.end
        13
        """
        sm = SM(string)
        capture_text(sm)
        string, slots = Utterance.stripped(string, sm.slots)
        return Utterance(string, slots)


class Slot(object):
    def __init__(self, name, entity, range, text, tag_range):
        self.name = name
        self.entity = entity
        self.range = range
        self.text = text
        self.tag_range = tag_range


class Text(object):
    def __init__(self, text, range):
        self.text = text
        self.range = range


class Range(object):
    def __init__(self, start, end=None):
        self.start = start
        self.end = end

    @property
    def size(self):
        return self.end - self.start + 1


class SM(object):
    """State Machine for parsing"""

    def __init__(self, input):
        self.input = input
        self.slots = []
        self.current = 0

    def add_slot(self, slot_start, name, entity):
        """Adds a named slot

        :param slot_start: int position where the slot tag started
        :param name: string name of the slot
        :param entity: string entity id
        """
        tag_range = Range(slot_start - 1)
        slot = Slot(name=name, entity=entity, range=None, text=None,
                    tag_range=tag_range)
        self.slots.append(slot)

    def add_text(self, text):
        """Adds a simple text slot using the current position

        :param text: text of the slot
        """
        start = self.current
        end = start + len(text)
        slot = Text(text=text, range=Range(start=start, end=end))
        self.slots.append(slot)

    def add_tagged(self, text):
        """Adds this text to the last slot

        :param text: string the text that was tagged
        """
        assert self.slots
        slot = self.slots[-1]
        slot.text = text
        slot.tag_range.end = self.current - 1
        slot.range = Range(start=self.current,
                           end=self.current + len(text))

    def find(self, s):
        return self.input.find(s, self.current)

    def move(self, pos):
        """Moves the cursor of the state to position after given

        :param pos: int position to place the cursor just after
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
            raise Exception("Bad key")


def capture_text(state):
    next_pos = state.find('[')
    sub = state[:] if next_pos < 0 else state[:next_pos]

    if sub.strip():
        state.add_text(sub)
    if next_pos >= 0:
        state.move(next_pos)
        capture_slot(state)


def capture_slot(state):
    slot_start = state.current
    next_pos = state.find(':')
    if next_pos < 0:
        raise BadFormat()
    else:
        slot_name = state[:next_pos]
        state.move(next_pos)
        next_pos = state.find(']')
        if next_pos < 0:
            raise BadFormat()
        entity = state[:next_pos]
        state.move(next_pos)
        state.add_slot(slot_start, slot_name, entity)
        if state.read() != '(':
            raise BadFormat()
        capture_tagged(state)


def capture_tagged(state):
    next_pos = state.find(')')
    if next_pos < 1:
        raise BadFormat()
    else:
        tagged_text = state[:next_pos]
        state.add_tagged(tagged_text)
        state.move(next_pos)
        capture_text(state)


class BadFormat(Exception):
    pass
