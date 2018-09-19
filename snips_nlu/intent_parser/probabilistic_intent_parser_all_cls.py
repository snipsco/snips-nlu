from __future__ import unicode_literals

from builtins import str

from snips_nlu.constants import RES_INTENT_NAME
from snips_nlu.intent_parser import ProbabilisticIntentParser
from snips_nlu.pipeline.configs import ProbabilisticIntentParserAllClsConfig
from snips_nlu.result import empty_result, parsing_result
from snips_nlu.utils import NotTrained


class ProbabilisticIntentParserAllCls(ProbabilisticIntentParser):
    unit_name = "probabilistic_intent_parser_all_cls"
    config_type = ProbabilisticIntentParserAllClsConfig

    def parse(self, text, intents=None):
        """Performs intent parsing on the provided *text* by first classifying
        the intent and then using the correspond slot filler to extract slots

        Args:
            text (str): Input
            intents (str or list of str): If provided, reduces the scope of
                intent parsing to the provided list of intents

        Returns:
            dict: The most likely intent along with the extracted slots. See
            :func:`.parsing_result` for the output format.

        Raises:
            NotTrained: When the intent parser is not fitted
        """
        if not self.fitted:
            raise NotTrained("ProbabilisticIntentParserAllCls must be fitted")

        if isinstance(intents, str):
            intents = [intents]

        intent_result = self.intent_classifier.get_intent(text, intents)
        if intent_result is None:
            return [empty_result(text)]

        results = []
        for entry in intent_result:
            intent_name = entry[RES_INTENT_NAME]
            if intent_name != "None":
                slots = self.slot_fillers[intent_name].get_slots(text)
            else:
                slots = []
            results.append(parsing_result(text, entry, slots))              
        return results
