from enum import Enum

from snips_nlu.result import ParsedSlot

BEGINNING_PREFIX = 'B-'
INSIDE_PREFIX = 'I-'
LAST_PREFIX = 'L-'
UNIT_PREFIX = 'U-'
OUTSIDE = 'O'


class Tagging(Enum):
    NONE = 0
    BIO = 1
    BILOU = 2


def is_other(label):
    return label == OUTSIDE or label is None


def labels_to_tags(labels, tagging):
    if tagging == Tagging.NONE:
        return [l if not is_other(l) else OUTSIDE for l in labels]
    if tagging == Tagging.BILOU:
        return labels_to_bilou_tags(labels)
    if tagging == Tagging.BIO:
        return labels_to_bio_tags(labels)
    raise ValueError("Invalid value for tagging: %s" % tagging)


def labels_to_bilou_tags(labels):
    bilou_labels = []
    if len(labels) == 1:
        if is_other(labels[0]):
            bilou_labels.append(OUTSIDE)
        else:
            bilou_labels.append(UNIT_PREFIX + labels[0])
        return bilou_labels

    for i in range(len(labels)):
        if is_other(labels[i]):
            bilou_labels.append(OUTSIDE)
        elif i == 0:
            bilou_labels.append(BEGINNING_PREFIX + labels[i])
        elif i == len(labels) - 1:
            if labels[i] == labels[i - 1]:
                bilou_labels.append(LAST_PREFIX + labels[i])
            else:
                bilou_labels.append(UNIT_PREFIX + labels[i])
        else:
            if labels[i] != labels[i - 1]:
                if labels[i] != labels[i + 1]:
                    bilou_labels.append(UNIT_PREFIX + labels[i])
                else:
                    bilou_labels.append(BEGINNING_PREFIX + labels[i])
            else:
                if labels[i] != labels[i + 1]:
                    bilou_labels.append(LAST_PREFIX + labels[i])
                else:
                    bilou_labels.append(INSIDE_PREFIX + labels[i])

    return bilou_labels


def labels_to_bio_tags(labels):
    bio_labels = []
    if len(labels) == 1:
        if is_other(labels[0]):
            bio_labels.append(OUTSIDE)
        else:
            bio_labels.append(BEGINNING_PREFIX + labels[0])
        return bio_labels

    for i in range(len(labels)):
        if is_other(labels[i]):
            bio_labels.append(OUTSIDE)
        elif i == 0:
            bio_labels.append(BEGINNING_PREFIX + labels[i])
        elif labels[i] != labels[i - 1]:
            bio_labels.append(BEGINNING_PREFIX + labels[i])
        else:
            bio_labels.append(INSIDE_PREFIX + labels[i])

    return bio_labels


def tags_to_labels(labels, tagging):
    if tagging is Tagging.NONE:
        return labels
    return [label[2:] if label != OUTSIDE else None for label in labels]


def remove_bilou_tags(bilou_labels):
    return [label if label == OUTSIDE else label[2:] for label in bilou_labels]


def build_parsed_entities(text, tokens, labels, slot_name_to_entity_mapping):
    slots = []
    last_slot_name = None
    for i in range(len(tokens)):
        token = tokens[i]
        slot_name = labels[i]
        if slot_name == OUTSIDE:
            last_slot_name = slot_name
            continue
        if slot_name == last_slot_name:
            last_range = slots[-1]['range']
            updated_range = (last_range[0], token.range[1])
            slots[-1].update({'range': updated_range})
        else:
            slot = {'slot_name': slot_name, 'range': token.range}
            slots.append(slot)
            last_slot_name = slot_name

    parsed_entities = []
    for slot in slots:
        rng = slot['range']
        value = text[rng[0]:rng[1]]
        slot_name = slot['slot_name']
        entity_name = slot_name_to_entity_mapping[slot_name]
        entity = ParsedSlot(rng, value, entity_name, slot_name)
        parsed_entities.append(entity)
    return parsed_entities
