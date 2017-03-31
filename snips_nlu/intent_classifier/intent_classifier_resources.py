import io
import os

from snips_nlu.utils import RESOURCES_PATH

STOP_WORDS = None
SUBTITLES = None


def get_stop_words(language):
    global STOP_WORDS
    if STOP_WORDS is None:
        stop_words_file_path = os.path.join(RESOURCES_PATH, '%s/stop_words.txt' % language.iso_code)
        with io.open(stop_words_file_path, encoding='utf8') as f:
            lines = [l.strip() for l in f]
            STOP_WORDS = set(l for l in lines if len(l) > 0)

    return STOP_WORDS


def get_subtitles(language):
    global SUBTITLES
    if SUBTITLES is None:
        subtitles_file_path = os.path.join(RESOURCES_PATH, '%s/subtitles.txt' % language.iso_code)
        with io.open(subtitles_file_path, encoding='utf8') as f:
            lines = [l.strip() for l in f]
            SUBTITLES = set(l for l in lines if len(l) > 0)

    return SUBTITLES
