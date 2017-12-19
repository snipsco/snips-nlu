from snips_nlu.intent_classifier.log_reg_classifier import \
    LogRegIntentClassifier
from snips_nlu.intent_parser.deterministic_intent_parser import \
    DeterministicIntentParser
from snips_nlu.intent_parser.probabilistic_intent_parser import \
    ProbabilisticIntentParser
from snips_nlu.nlu_engine.nlu_engine import SnipsNLUEngine
from snips_nlu.slot_filler.crf_slot_filler import CRFSlotFiller

_NLU_PROCESSING_UNITS = [SnipsNLUEngine, ProbabilisticIntentParser,
                         DeterministicIntentParser, LogRegIntentClassifier,
                         CRFSlotFiller]

NLU_PROCESSING_UNITS = {
    unit.unit_name: unit for unit in _NLU_PROCESSING_UNITS
}


def register_processing_unit(unit_type):
    if unit_type.unit_name not in NLU_PROCESSING_UNITS:
        NLU_PROCESSING_UNITS[unit_type.unit_name] = unit_type


def reset_processing_units():
    global NLU_PROCESSING_UNITS
    NLU_PROCESSING_UNITS = {
        unit.unit_name: unit for unit in _NLU_PROCESSING_UNITS
    }
