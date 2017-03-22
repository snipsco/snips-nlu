from snips_nlu.result import ParsedEntity

BEGINNING_PREFIX = 'B-'
INSIDE_PREFIX = 'I-'
LAST_PREFIX = 'L-'
UNIT_PREFIX = 'U-'
OUTSIDE = 'O'


def is_other(label):
    return label == OUTSIDE or label is None


def add_bilou_tags(labels):
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
        entity = ParsedEntity(rng, value, entity_name, slot_name)
        parsed_entities.append(entity)
    return parsed_entities
