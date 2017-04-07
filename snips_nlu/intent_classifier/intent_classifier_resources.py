import io
import os

from snips_nlu.utils import get_resources_path

STOP_WORDS = None
SUBTITLES = None


def get_stop_words(language):
    global STOP_WORDS
    if STOP_WORDS is None:
        stop_words_file_path = os.path.join(get_resources_path(language),
                                            'stop_words.txt')
        with io.open(stop_words_file_path, encoding='utf8') as f:
            lines = [l.strip() for l in f]
            STOP_WORDS = set(l for l in lines if len(l) > 0)

    return STOP_WORDS


def get_subtitles(language):
    global SUBTITLES
    if SUBTITLES is None:
        subtitles_file_path = os.path.join(get_resources_path(language),
                                           'subtitles.txt')
        with io.open(subtitles_file_path, encoding='utf8') as f:
            lines = [l.strip() for l in f]
            SUBTITLES = set(l for l in lines if len(l) > 0)

    return SUBTITLES
