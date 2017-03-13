BEGINNING_PREFIX = 'B-'
INSIDE_PREFIX = 'I-'
LAST_PREFIX = 'L-'
UNIT_PREFIX = 'U-'
OUTSIDE = 'O'


def is_other(label):
    return label == OUTSIDE or label is None


def get_bilou_labels(labels):
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
