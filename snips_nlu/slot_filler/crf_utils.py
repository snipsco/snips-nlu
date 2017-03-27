from enum import Enum, unique

from snips_nlu.tokenization import tokenize, Token

BEGINNING_PREFIX = 'B-'
INSIDE_PREFIX = 'I-'
LAST_PREFIX = 'L-'
UNIT_PREFIX = 'U-'
OUTSIDE = 'O'


@unique
class Tagging(Enum):
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
                "range": (tokens[current_slot_start].start, tokens[i].end),
                "slot_name": tag_name_to_slot_name(tag)
            })
    return slots


def bilou_tags_to_slots(tags, tokens):
    slots = []
    current_slot_start = 0
    for i, tag in enumerate(tags):
        if tag.startswith(UNIT_PREFIX):
            slots.append({"range": (tokens[i].start, tokens[i].end),
                          "slot_name": tag_name_to_slot_name(tag)})
        if tag.startswith(BEGINNING_PREFIX):
            current_slot_start = i
        if tag.startswith(LAST_PREFIX):
            slots.append({"range": (tokens[current_slot_start].start,
                                    tokens[i].end),
                          "slot_name": tag_name_to_slot_name(tag)})
    return slots


def io_tags_to_slots(tags, tokens):
    slots = []
    current_slot_start = None
    for i, tag in enumerate(tags):
        if tag == OUTSIDE:
            if current_slot_start is not None:
                slots.append({
                    "range": (tokens[current_slot_start].start,
                              tokens[i - 1].end),
                    "slot_name": tag_name_to_slot_name(tag)
                })
                current_slot_start = None
        else:
            if current_slot_start is None:
                current_slot_start = i

    if current_slot_start is not None:
        slots.append({
            "range": (tokens[current_slot_start].start,
                      tokens[len(tokens) - 1].end),
            "slot_name": tag_name_to_slot_name(tags[-1])
        })
    return slots


def tags_to_slots(tokens, tags, tagging):
    if tagging == Tagging.IO:
        return io_tags_to_slots(tags, tokens)
    elif tagging == Tagging.BIO:
        return bio_tags_to_slots(tags, tokens)
    elif tagging == Tagging.BILOU:
        return bilou_tags_to_slots(tags, tokens)
    else:
        raise ValueError("Unknown tagging %s" % tagging)


def positive_tagging(tagging, slot_name, slot_size):
    if tagging == Tagging.IO:
        tags = [INSIDE_PREFIX + slot_name for _ in xrange(slot_size)]
    elif tagging == Tagging.BIO:
        tags = [BEGINNING_PREFIX + slot_name]
        tags += [INSIDE_PREFIX + slot_name for _ in xrange(1, slot_size)]
    elif tagging == Tagging.BILOU:
        if slot_size == 1:
            tags = [UNIT_PREFIX + slot_name]
        else:
            tags = [BEGINNING_PREFIX + slot_name]
            tags += [INSIDE_PREFIX + slot_name
                     for _ in xrange(1, slot_size - 1)]
            tags.append(LAST_PREFIX + slot_name)
    else:
        raise ValueError("Invalid tagging %s" % tagging)
    return tags


def negative_tagging(size):
    return [OUTSIDE for _ in xrange(size)]


def utterance_to_sample(query_data, tagging):
    tokens, tags = [], []
    current_length = 0
    # tokens = tokenize("".join(q["text"] for q in query_data))
    for i, chunk in enumerate(query_data):
        chunk_tokens = tokenize(chunk["text"])
        tokens += [Token(t.value, current_length + t.start,
                         current_length + t.end) for t in chunk_tokens]
        current_length += len(chunk["text"])
        if "slot_name" not in chunk:
            tags += negative_tagging(len(chunk_tokens))
        else:
            tags += positive_tagging(tagging, chunk["slot_name"],
                                     len(chunk_tokens))
    return {"tokens": tokens, "tags": tags}
