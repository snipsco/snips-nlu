import glob
import io
import os
import ujson

from snips_nlu.utils import RESOURCES_PATH

_LANGUAGE_STEMS = None


def language_stems(language):
    global _LANGUAGE_STEMS
    if _LANGUAGE_STEMS is None:
        _LANGUAGE_STEMS = dict()
        stems_paths = glob.glob(os.path.join(RESOURCES_PATH, "stems_*.json"))
        for path in stems_paths:
            _, filename = os.path.split(path)
            lang = os.path.splitext(filename)[0].split("_")[-1]
            with io.open(path, encoding="utf8") as f:
                _LANGUAGE_STEMS[lang] = ujson.load(f)

    return _LANGUAGE_STEMS[language.iso_code]


def stem(string, language, *default):
    if len(default) == 0:
        return language_stems(language)[string]
    elif len(default) == 1:
        return language_stems(language).get(string, default[0])
    else:
        raise ValueError("Can only have 0 or 1 default argument")
