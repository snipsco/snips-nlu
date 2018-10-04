from __future__ import unicode_literals

from snips_nlu.constants import (
    LANGUAGE_DE, LANGUAGE_EN, LANGUAGE_ES, LANGUAGE_FR, LANGUAGE_IT,
    LANGUAGE_JA, LANGUAGE_KO)
from .config_de import CONFIG as CONFIG_DE
from .config_en import CONFIG as CONFIG_EN
from .config_es import CONFIG as CONFIG_ES
from .config_fr import CONFIG as CONFIG_FR
from .config_it import CONFIG as CONFIG_IT
from .config_ja import CONFIG as CONFIG_JA
from .config_ko import CONFIG as CONFIG_KO

DEFAULT_CONFIGS = {
    LANGUAGE_DE: CONFIG_DE,
    LANGUAGE_EN: CONFIG_EN,
    LANGUAGE_ES: CONFIG_ES,
    LANGUAGE_FR: CONFIG_FR,
    LANGUAGE_IT: CONFIG_IT,
    LANGUAGE_JA: CONFIG_JA,
    LANGUAGE_KO: CONFIG_KO,
}
