from snips_nlu.intent_classifier import LogRegIntentClassifier
from snips_nlu.intent_parser import (DeterministicIntentParser,
                                     ProbabilisticIntentParser)
from snips_nlu.nlu_engine.nlu_engine import SnipsNLUEngine
from snips_nlu.slot_filler import CRFSlotFiller

BUILTIN_NLU_PROCESSING_UNITS = [
    SnipsNLUEngine,
    ProbabilisticIntentParser,
    DeterministicIntentParser,
    LogRegIntentClassifier,
    CRFSlotFiller
]

NLU_PROCESSING_UNITS = {
    unit.unit_name: unit for unit in BUILTIN_NLU_PROCESSING_UNITS
}


def register_processing_unit(unit_type):
    """Allow to register a custom processing unit"""
    if unit_type.unit_name not in NLU_PROCESSING_UNITS:
        NLU_PROCESSING_UNITS[unit_type.unit_name] = unit_type


def reset_processing_units():
    """Remove all the custom processing units and reset to the default ones as
        defined in *BUILTIN_NLU_PROCESSING_UNITS*"""
    global NLU_PROCESSING_UNITS
    NLU_PROCESSING_UNITS = {
        unit.unit_name: unit for unit in BUILTIN_NLU_PROCESSING_UNITS
    }
