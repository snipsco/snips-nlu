from __future__ import unicode_literals

from builtins import str, zip

from snips_nlu.intent_classifier import LogRegIntentClassifier
from snips_nlu.pipeline.configs import LogRegIntentClassifierAllClsConfig
from snips_nlu.result import intent_classification_result
from snips_nlu.utils import NotTrained
from snips_nlu.intent_classifier.log_reg_classifier_utils import (
    build_training_data, get_regularization_factor, text_to_utterance)

class LogRegIntentClassifierAllCls(LogRegIntentClassifier):
    unit_name = "log_reg_intent_classifier_all_cls"
    config_type = LogRegIntentClassifierAllClsConfig

    def get_intent(self, text, intents_filter=None):
        """Performs intent classification on the provided *text*

        Args:
            text (str): Input
            intents_filter (str or list of str): When defined, it will find
                the most likely intent among the list, otherwise it will use
                the whole list of intents defined in the dataset

        Returns:
            list(dict): All intents in decreasing probability

        Raises:
            NotTrained: When the intent classifier is not fitted

        """
        if not self.fitted:
            raise NotTrained('LogRegIntentClassifier must be fitted')

        if isinstance(intents_filter, str):
            intents_filter = [intents_filter]

        if not text or not self.intent_list \
                or self.featurizer is None or self.classifier is None:
            return None

        if len(self.intent_list) == 1:
            if self.intent_list[0] is None:
                return None
            return intent_classification_result(self.intent_list[0], 1.0)

        X = self.featurizer.transform([text_to_utterance(text)])  # pylint: disable=C0103
        proba_vec = self.classifier.predict_proba(X)[0]
        intents_probas = sorted(zip(self.intent_list, proba_vec),
                                key=lambda p: -p[1])
        results = []
        for intent, proba in intents_probas:
            if intent is None:
                results.append(intent_classification_result("None", proba))
                continue
            if intents_filter is None or intent in intents_filter:
                results.append(intent_classification_result(intent, proba))
        return results
