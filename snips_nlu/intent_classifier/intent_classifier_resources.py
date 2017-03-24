import io
import os
import re

from snips_nlu.utils import RESOURCES_PATH

EN_STOP_WORDS = None
EN_SUBTITLES = None

def get_stop_words(language='en'):
    global EN_STOP_WORDS
    if EN_STOP_WORDS is None:
        with io.open(os.path.join(RESOURCES_PATH, '%s_stop_words.txt' %language), encoding='utf8') as f:            
            lines = [ l.strip() for l in f ]
            EN_STOP_WORDS = set( l for l in lines if len(l) > 0 )

    return EN_STOP_WORDS

def get_subtitles(language='en'):
    global EN_SUBTITLES
    if EN_SUBTITLES is None:
        with io.open(os.path.join(RESOURCES_PATH, '%s_subtitles.txt' %language), encoding='utf8') as f:            
            lines = [ l.strip() for l in f ]
            EN_SUBTITLES = set( l for l in lines if len(l) > 0 )

    return EN_SUBTITLES

