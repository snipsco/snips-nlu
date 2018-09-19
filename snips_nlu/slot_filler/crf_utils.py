from __future__ import unicode_literals
import types

from builtins import range
from enum import Enum, unique

from snips_nlu.constants import END, SLOT_NAME, START, TEXT, PROBABILITY
from snips_nlu.preprocessing import Token, tokenize
from snips_nlu.result import unresolved_slot

BEGINNING_PREFIX = "B-"
INSIDE_PREFIX = "I-"
LAST_PREFIX = "L-"
UNIT_PREFIX = "U-"
OUTSIDE = "O"

RANGE = "range"
TAGS = "tags"
TOKENS = "tokens"


@unique
class TaggingScheme(Enum):
    """CRF Coding Scheme"""

    IO = 0
    """Inside-Outside scheme"""
    BIO = 1
    """Beginning-Inside-Outside scheme"""
    BILOU = 2
    """Beginning-Inside-Last-Outside-Unit scheme, sometimes referred as
         BWEMO"""


         
def _resolve_slot_name(tag_tuple):
    if (isinstance(tag_tuple[0], types.TupleType)):
        return tag_tuple[0][0]          
    else:
        return tag_tuple[0]


def tag_name_to_slot_name(tag):
    return _resolve_slot_name(tag)[2:]


def start_of_io_slot(tags, i):
    if i == 0:
        return _resolve_slot_name(tags[i]) != OUTSIDE
    if _resolve_slot_name(tags[i]) == OUTSIDE:
        return False
    return _resolve_slot_name(tags[i - 1]) == OUTSIDE


def end_of_io_slot(tags, i):
    if i + 1 == len(tags):
        return _resolve_slot_name(tags[i]) != OUTSIDE
    if _resolve_slot_name(tags[i]) == OUTSIDE:
        return False
    return _resolve_slot_name(tags[i + 1]) == OUTSIDE


def start_of_bio_slot(tags, i):
    if i == 0:
        return _resolve_slot_name(tags[i]) != OUTSIDE
    if _resolve_slot_name(tags[i]) == OUTSIDE:
        return False
    try:    
        if _resolve_slot_name(tags[i]).startswith(BEGINNING_PREFIX):
            return True
    except:
        raise ValueError(tags)       
    if _resolve_slot_name(tags[i - 1]) != OUTSIDE:
        return False
    return True


def end_of_bio_slot(tags, i):
    if i + 1 == len(tags):
        return _resolve_slot_name(tags[i]) != OUTSIDE
    if _resolve_slot_name(tags[i]) == OUTSIDE:
        return False
    if _resolve_slot_name(tags[i + 1]).startswith(INSIDE_PREFIX):
        return False
    return True


def start_of_bilou_slot(tags, i):
    if i == 0:
        return _resolve_slot_name(tags[i]) != OUTSIDE
    if _resolve_slot_name(tags[i]) == OUTSIDE:
        return False
    if _resolve_slot_name(tags[i]).startswith(BEGINNING_PREFIX):
        return True
    if _resolve_slot_name(tags[i]).startswith(UNIT_PREFIX):
        return True
    if _resolve_slot_name(tags[i - 1]).startswith(UNIT_PREFIX):
        return True
    if _resolve_slot_name(tags[i - 1]).startswith(LAST_PREFIX):
        return True
    if _resolve_slot_name(tags[i - 1]) != OUTSIDE:
        return False
    return True


def end_of_bilou_slot(tags, i):
    if i + 1 == len(tags):
        return _resolve_slot_name(tags[i]) != OUTSIDE
    if _resolve_slot_name(tags[i]) == OUTSIDE:
        return False
    if _resolve_slot_name(tags[i + 1]) == OUTSIDE:
        return True
    if _resolve_slot_name(tags[i]).startswith(LAST_PREFIX):
        return True
    if _resolve_slot_name(tags[i]).startswith(UNIT_PREFIX):
        return True
    if _resolve_slot_name(tags[i + 1]).startswith(BEGINNING_PREFIX):
        return True
    if _resolve_slot_name(tags[i + 1]).startswith(UNIT_PREFIX):
        return True
    return False


def list_product(list):
    from operator import mul
    return reduce(mul, list, 1)


def _tags_to_preslots(tags, tokens, is_start_of_slot, is_end_of_slot, with_probs = False):
    slots = []
    current_slot_start = 0
    for i, tag in enumerate(tags):
        tag_ = {}  
        if is_start_of_slot(tags, i):
            current_slot_start = i
        if is_end_of_slot(tags, i):
            tag_[RANGE] = {
                    START: tokens[current_slot_start].start,
                    END: tokens[i].end
                           }
            tag_[SLOT_NAME] = tag_name_to_slot_name(tag)      
            if with_probs:
                tag_[PROBABILITY] = list_product([tag[1] for tag in tags[current_slot_start:i+1]])
            slots.append(tag_)
            current_slot_start = i
    return slots


def tags_to_preslots(tokens, tags, tagging_scheme, with_probs = False):
    if tagging_scheme == TaggingScheme.IO:
        slots = _tags_to_preslots(tags, tokens, start_of_io_slot,
                                  end_of_io_slot, with_probs = with_probs)
    elif tagging_scheme == TaggingScheme.BIO:
        slots = _tags_to_preslots(tags, tokens, start_of_bio_slot,
                                  end_of_bio_slot, with_probs = with_probs)
    elif tagging_scheme == TaggingScheme.BILOU:
        slots = _tags_to_preslots(tags, tokens, start_of_bilou_slot,
                                  end_of_bilou_slot, with_probs = with_probs)
    else:
        raise ValueError("Unknown tagging scheme %s" % tagging_scheme)
    return slots


def tags_to_slots(text, tokens, tags, tagging_scheme, intent_slots_mapping, with_probs = False):
    slots = tags_to_preslots(tokens, tags, tagging_scheme, with_probs = with_probs)
    temp_slots = []
    for slot in slots: 
        slot_params = dict(unresolved_slot(range=slot[RANGE],
                    value=text[slot[RANGE][START]:slot[RANGE][END]],
                    entity=intent_slots_mapping[slot[SLOT_NAME]],
                    slot_name=slot[SLOT_NAME]))
        if with_probs:
            slot_params['probability'] = slot[PROBABILITY]
        temp_slots.append(unresolved_slot(**slot_params))
    return temp_slots      


def positive_tagging(tagging_scheme, slot_name, slot_size):
    if slot_name == OUTSIDE:
        return [OUTSIDE for _ in range(slot_size)]

    if tagging_scheme == TaggingScheme.IO:
        tags = [INSIDE_PREFIX + slot_name for _ in range(slot_size)]
    elif tagging_scheme == TaggingScheme.BIO:
        if slot_size > 0:
            tags = [BEGINNING_PREFIX + slot_name]
            tags += [INSIDE_PREFIX + slot_name for _ in range(1, slot_size)]
        else:
            tags = []
    elif tagging_scheme == TaggingScheme.BILOU:
        if slot_size == 0:
            tags = []
        elif slot_size == 1:
            tags = [UNIT_PREFIX + slot_name]
        else:
            tags = [BEGINNING_PREFIX + slot_name]
            tags += [INSIDE_PREFIX + slot_name
                     for _ in range(1, slot_size - 1)]
            tags.append(LAST_PREFIX + slot_name)
    else:
        raise ValueError("Invalid tagging scheme %s" % tagging_scheme)
    return tags


def negative_tagging(size):
    return [OUTSIDE for _ in range(size)]


def utterance_to_sample(query_data, tagging_scheme, language):
    tokens, tags = [], []
    current_length = 0
    for chunk in query_data:
        chunk_tokens = tokenize(chunk[TEXT], language)
        tokens += [Token(t.value, current_length + t.start,
                         current_length + t.end) for t in chunk_tokens]
        current_length += len(chunk[TEXT])
        if SLOT_NAME not in chunk:
            tags += negative_tagging(len(chunk_tokens))
        else:
            tags += positive_tagging(tagging_scheme, chunk[SLOT_NAME],
                                     len(chunk_tokens))
    return {TOKENS: tokens, TAGS: tags}


def get_scheme_prefix(index, indexes, tagging_scheme):
    if tagging_scheme == TaggingScheme.IO:
        return INSIDE_PREFIX
    elif tagging_scheme == TaggingScheme.BIO:
        if index == indexes[0]:
            return BEGINNING_PREFIX
        return INSIDE_PREFIX
    elif tagging_scheme == TaggingScheme.BILOU:
        if len(indexes) == 1:
            return UNIT_PREFIX
        if index == indexes[0]:
            return BEGINNING_PREFIX
        if index == indexes[-1]:
            return LAST_PREFIX
        return INSIDE_PREFIX
    else:
        raise ValueError("Invalid tagging scheme %s" % tagging_scheme)
