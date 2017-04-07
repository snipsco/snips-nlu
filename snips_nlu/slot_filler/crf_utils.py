from enum import Enum, unique

from snips_nlu.constants import TEXT, SLOT_NAME
from snips_nlu.tokenization import tokenize, Token

BEGINNING_PREFIX = 'B-'
INSIDE_PREFIX = 'I-'
LAST_PREFIX = 'L-'
UNIT_PREFIX = 'U-'
OUTSIDE = 'O'

RANGE = "range"
TAGS = "tags"
TOKENS = "tokens"


@unique
class TaggingScheme(Enum):
    IO = 0
    BIO = 1
    BILOU = 2


def tag_name_to_slot_name(tag):
    return tag[2:]


def end_of_boi_slot(tags, i):
    if i + 1 == len(tags):
        return tags[i] != OUTSIDE
    else:
        if tags[i] == OUTSIDE:
            return False
        else:
            if tags[i + 1].startswith(INSIDE_PREFIX):
                return False
            else:
                return True


def bio_tags_to_slots(tags, tokens):
    slots = []
    current_slot_start = 0
    for i, tag in enumerate(tags):
        if tag.startswith(BEGINNING_PREFIX):
            current_slot_start = i
        if end_of_boi_slot(tags, i):
            slots.append({
                RANGE: (tokens[current_slot_start].start, tokens[i].end),
                SLOT_NAME: tag_name_to_slot_name(tag)
            })
    return slots


def bilou_tags_to_slots(tags, tokens):
    slots = []
    current_slot_start = 0
    for i, tag in enumerate(tags):
        if tag.startswith(UNIT_PREFIX):
            slots.append({RANGE: (tokens[i].start, tokens[i].end),
                          SLOT_NAME: tag_name_to_slot_name(tag)})
        if tag.startswith(BEGINNING_PREFIX):
            current_slot_start = i
        if tag.startswith(LAST_PREFIX):
            slots.append({RANGE: (tokens[current_slot_start].start,
                                  tokens[i].end),
                          SLOT_NAME: tag_name_to_slot_name(tag)})
    return slots


def io_tags_to_slots(tags, tokens):
    slots = []
    current_slot_start = None
    for i, tag in enumerate(tags):
        if tag == OUTSIDE:
            if current_slot_start is not None:
                slots.append({
                    RANGE: (tokens[current_slot_start].start,
                            tokens[i - 1].end),
                    SLOT_NAME: tag_name_to_slot_name(tag)
                })
                current_slot_start = None
        else:
            if current_slot_start is None:
                current_slot_start = i

    if current_slot_start is not None:
        slots.append({
            RANGE: (tokens[current_slot_start].start,
                    tokens[len(tokens) - 1].end),
            SLOT_NAME: tag_name_to_slot_name(tags[-1])
        })
    return slots


def tags_to_slots(tokens, tags, tagging_scheme):
    if tagging_scheme == TaggingScheme.IO:
        return io_tags_to_slots(tags, tokens)
    elif tagging_scheme == TaggingScheme.BIO:
        return bio_tags_to_slots(tags, tokens)
    elif tagging_scheme == TaggingScheme.BILOU:
        return bilou_tags_to_slots(tags, tokens)
    else:
        raise ValueError("Unknown tagging scheme %s" % tagging_scheme)


def positive_tagging(tagging_scheme, slot_name, slot_size):
    if tagging_scheme == TaggingScheme.IO:
        tags = [INSIDE_PREFIX + slot_name for _ in xrange(slot_size)]
    elif tagging_scheme == TaggingScheme.BIO:
        tags = [BEGINNING_PREFIX + slot_name]
        tags += [INSIDE_PREFIX + slot_name for _ in xrange(1, slot_size)]
    elif tagging_scheme == TaggingScheme.BILOU:
        if slot_size == 1:
            tags = [UNIT_PREFIX + slot_name]
        else:
            tags = [BEGINNING_PREFIX + slot_name]
            tags += [INSIDE_PREFIX + slot_name
                     for _ in xrange(1, slot_size - 1)]
            tags.append(LAST_PREFIX + slot_name)
    else:
        raise ValueError("Invalid tagging scheme %s" % tagging_scheme)
    return tags


def negative_tagging(size):
    return [OUTSIDE for _ in xrange(size)]


def utterance_to_sample(query_data, tagging_scheme):
    tokens, tags = [], []
    current_length = 0
    for i, chunk in enumerate(query_data):
        chunk_tokens = tokenize(chunk[TEXT])
        tokens += [Token(t.value, current_length + t.start,
                         current_length + t.end) for t in chunk_tokens]
        current_length += len(chunk[TEXT])
        if SLOT_NAME not in chunk:
            tags += negative_tagging(len(chunk_tokens))
        else:
            tags += positive_tagging(tagging_scheme, chunk[SLOT_NAME],
                                     len(chunk_tokens))
    return {TOKENS: tokens, TAGS: tags}
