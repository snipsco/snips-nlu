import io
import os

from snips_nlu.utils import get_resources_path

STOP_WORDS = dict()
SUBTITLES = dict()


def get_stop_words(language):
    global STOP_WORDS
    if language.iso_code not in STOP_WORDS:
        stop_words_file_path = os.path.join(
            get_resources_path(language), 'stop_words.txt')
        with io.open(stop_words_file_path, encoding='utf8') as f:
            lines = [l.strip() for l in f]
            STOP_WORDS[language.iso_code] = set(l for l in lines if len(l) > 0)

    return STOP_WORDS[language.iso_code]


def get_subtitles(language):
    global SUBTITLES
    if language.iso_code not in SUBTITLES:
        subtitles_file_path = os.path.join(
            get_resources_path(language), 'subtitles.txt')
        with io.open(subtitles_file_path, encoding='utf8') as f:
            lines = [l.strip() for l in f]
            SUBTITLES[language.iso_code] = set(l for l in lines if len(l) > 0)

    return SUBTITLES[language.iso_code]
