# coding=utf-8
from __future__ import division, unicode_literals

import itertools
import json
from copy import deepcopy

import fire
import matplotlib.pyplot as plt
import numpy as np
from future.utils import iteritems
from pathlib import Path


def get_f1(metrics):
    metrics = deepcopy(metrics)
    not_none_f1s = [i["intent"]["f1"] for name, i
                    in iteritems(metrics["metrics"]) if name != "null"]
    return sum(not_none_f1s) / len(not_none_f1s)


def draw_confusion_matrix(metrics, metrics_path, f1):
    cm = np.array(metrics["confusion_matrix"]["matrix"])
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    classes = metrics["confusion_matrix"]["intents"]

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix, F1: %s" % f1)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.tight_layout()

    metrics_path = Path(metrics_path)

    fig_path = Path(metrics_path).with_name(
        "%s_confusion_matrics.png" % metrics_path.stem)

    # fig = plt.gcf()
    # fig.set_size_inches(100, 100)

    plt.savefig(str(fig_path), dpi=300)


def aggregate_metrics(metrics_path):
    metrics_path = Path(metrics_path)
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    aggregated_intent_f1 = get_f1(metrics)

    draw_confusion_matrix(metrics, metrics_path, aggregated_intent_f1)


if __name__ == '__main__':
    fire.Fire(aggregate_metrics)
